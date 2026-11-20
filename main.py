"""Telegram bot that transcribes voice and audio messages with Groq Whisper and polishes the text."""

import asyncio
import io
import logging
import os
from typing import Awaitable, Callable

from dotenv import load_dotenv
from groq import AsyncGroq, BadRequestError, RateLimitError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

MODEL = "whisper-large-v3"
POLISH_MODEL = "openai/gpt-oss-120b"
MAX_FILE_SIZE = 20 * 1024 * 1024  # Telegram Bot API download limit for bots
TELEGRAM_MESSAGE_LIMIT = 4096
MAX_RETRY_AFTER = 30  # seconds; longer waits are reported to the user instead

POLISH_SYSTEM_PROMPT = """You are an expert editor specializing in cleaning and correcting speech-to-text (ASR/STT) transcriptions. Your task is to fix technical voice recognition errors and structure the text for readability while keeping the speaker's original style, slang, language, and voice completely intact.

Editing rules:
1. Spelling and reconstruction: Fix transcription errors, cut-off words, or phonetically misinterpreted words based on sentence context (e.g., reconstructing clipped characters or missing accents/diacritics). Keep the output in the exact same language as the input transcription.
2. Punctuation and formatting: Add punctuation marks (commas, periods, question/exclamation marks, ellipses, and em-dashes) that reflect the natural pauses, interruptions, and rhythm of speech. Break the text into paragraphs whenever there is a clear shift in topic or idea.
3. Strict fidelity: DO NOT summarize, DO NOT paraphrase, DO NOT replace colloquial vocabulary with formal language, and DO NOT remove characteristic filler words (e.g., "like", "you know", "o sea", "pues"), unless they are accidental syllable stutters caused by an audio glitch.

Output constraint:
Return ONLY the corrected text. Do not include greetings, introductions, wrapping quotation marks, footnotes, or explanations of the changes made."""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def require_env(name: str) -> str:
    """Return a required environment variable or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing {name}. Add it to the .env file (see .env.example).")
    return value


def parse_user_ids(raw: str) -> set[int]:
    """Parse a comma-separated list of Telegram user IDs."""
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            raise SystemExit(f"Invalid user ID in ALLOWED_USER_IDS: {part!r}")
    return ids


ALLOWED_USER_IDS = parse_user_ids(os.environ.get("ALLOWED_USER_IDS", ""))


def is_allowed(user_id: int) -> bool:
    """An empty allowlist means everyone is allowed."""
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def deny_access(update: Update) -> None:
    """Reply to a user who is not on the allowlist."""
    user_id = update.effective_user.id if update.effective_user else "unknown"
    await update.effective_message.reply_text(
        f"This bot is private. Your user ID is {user_id} — "
        "ask the bot owner to add it to ALLOWED_USER_IDS."
    )


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text into chunks that fit in a single Telegram message."""
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


def retry_after_seconds(error: RateLimitError) -> int:
    """Extract the retry-after delay (in seconds) from a Groq 429 response."""
    try:
        return max(int(error.response.headers.get("retry-after", "")), 1)
    except ValueError:
        return 60


async def retry_rate_limit(call: Callable[[], Awaitable[str]]) -> str:
    """Run an async Groq call, retrying once when rate limited."""
    for attempt in range(2):
        try:
            return await call()
        except RateLimitError as error:
            wait = retry_after_seconds(error)
            if attempt == 1 or wait > MAX_RETRY_AFTER:
                raise
            logger.warning("Groq rate limit hit, retrying in %ss", wait)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")  # the loop always returns or raises


async def transcribe_audio(groq: AsyncGroq, filename: str, data: bytes) -> str:
    """Transcribe audio with Groq, retrying once when rate limited."""

    async def call() -> str:
        result = await groq.audio.transcriptions.create(
            file=(filename, data),
            model=MODEL,
            response_format="json",
        )
        return result.text

    return await retry_rate_limit(call)


def polish_user_prompt(text: str) -> str:
    """Build the user message that asks for a cleaned-up transcription."""
    return (
        "Clean and format the following transcription:\n\n"
        f"<transcription>\n{text}\n</transcription>"
    )


async def polish_transcription(groq: AsyncGroq, text: str) -> str:
    """Polish a raw transcription with Groq chat completions, retrying once when rate limited."""

    async def call() -> str:
        completion = await groq.chat.completions.create(
            model=POLISH_MODEL,
            messages=[
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": polish_user_prompt(text)},
            ],
            max_completion_tokens=65536,  # model maximum; the API default could truncate long texts
            temperature=0,  # deterministic: keep edits conservative and close to the original wording
        )
        return (completion.choices[0].message.content or "").strip()

    return await retry_rate_limit(call)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None or not is_allowed(user.id):
        await deny_access(update)
        return
    await update.message.reply_text(
        "Send me a voice message or an audio file and I'll transcribe and polish it.\n"
        f"Your user ID: {user.id}"
    )


async def transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or not is_allowed(update.effective_user.id):
        await deny_access(update)
        return

    message = update.effective_message
    media = message.voice or message.audio

    if media.file_size and media.file_size > MAX_FILE_SIZE:
        await message.reply_text("That file is too large. Please send an audio of up to 20 MB.")
        return

    status = await message.reply_text("Transcribing...")

    try:
        tg_file = await media.get_file()
        buffer = io.BytesIO()
        await tg_file.download_to_memory(out=buffer)
        data = buffer.getvalue()

        if message.voice:
            filename = "voice.ogg"
        else:
            subtype = (media.mime_type or "audio/ogg").partition("/")[2] or "ogg"
            filename = media.file_name or f"audio.{subtype}"

        groq: AsyncGroq = context.application.bot_data["groq"]
        raw = (await transcribe_audio(groq, filename, data)).strip()

        if not raw:
            await status.edit_text("No speech detected.")
            return

        await status.edit_text("Polishing...")
        try:
            polished = await polish_transcription(groq, raw)
        except Exception:
            logger.exception("Polishing failed, sending the raw transcription")
            polished = ""

        chunks = split_message(polished or raw)
        await status.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await message.reply_text(chunk)
    except RateLimitError as error:
        wait = retry_after_seconds(error)
        await status.edit_text(f"The transcription service is busy. Try again in {wait}s.")
    except BadRequestError:
        await status.edit_text(
            "I couldn't process that file. Send a voice message or an audio file "
            "(ogg, mp3, m4a, wav, ...)."
        )
    except Exception:
        logger.exception("Transcription failed")
        await status.edit_text("Transcription failed. Please try again later.")


def main() -> None:
    """Start the bot and keep it running."""
    telegram_token = require_env("TELEGRAM_BOT_TOKEN")
    groq_api_key = require_env("GROQ_API_KEY")

    application = Application.builder().token(telegram_token).build()
    application.bot_data["groq"] = AsyncGroq(api_key=groq_api_key)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, transcribe))

    logger.info("Bot started, polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
