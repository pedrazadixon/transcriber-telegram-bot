# Telegram Transcriber Bot

A Telegram bot that receives voice messages and audio files, transcribes them with
[Groq](https://console.groq.com/docs/speech-to-text) (`whisper-large-v3`), polishes the result with
`openai/gpt-oss-120b`, and replies with the cleaned-up text.

## Setup

### 1. Create the bot with BotFather

1. Open Telegram and start a chat with [@BotFather](https://t.me/BotFather).
2. Send `/newbot`.
3. Choose a **display name** for the bot (e.g. `My Transcriber`).
4. Choose a **username** that ends in `bot` (e.g. `my_transcriber_bot`).
5. BotFather replies with an **HTTP API token** — copy it (looks like `123456789:AAE...`).

### 2. Get a Groq API key

1. Go to <https://console.groq.com> and sign in.
2. Open **API Keys** and create a new key.
3. Copy the key (starts with `gsk_`).

### 3. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in both values:

```dotenv
TELEGRAM_BOT_TOKEN=123456789:AAE...
GROQ_API_KEY=gsk_...
```

### 4. (Optional) Restrict access

By default anyone can use the bot. To allow only certain users:

1. Send `/start` to the bot — the reply shows your **Telegram user ID** (a number).
2. Add it to `.env`, comma-separated for several users:

```dotenv
ALLOWED_USER_IDS=123456789,987654321
```

3. Restart the bot. Anyone else gets a "This bot is private" reply that includes their own
   user ID, so they can send it to you to request access.

## Run

```bash
uv run main.py
```

The program starts and keeps running, polling Telegram for updates. Open your bot in Telegram and
send it a voice message or an audio file — it replies with the polished transcription.

## Usage

- Accepts **voice messages** and **audio files** (ogg, mp3, m4a, wav, ...) up to **20 MB**.
- The transcription language is detected automatically.
- The text is polished automatically: recognition errors are fixed and punctuation/paragraphs are
  added, while keeping the speaker's wording, slang, and style.
- Long transcriptions are split into multiple messages.
- `/start` shows a short usage hint.

## Notes

- Transcription runs on `whisper-large-v3` via the Groq API.
- Polishing runs on `openai/gpt-oss-120b` via the Groq chat API with a fixed editor prompt. If the
  polish step fails, the raw transcription is sent instead.
- Groq rate limits (developer tier): about **20 requests/min** and **2,000 requests/day** — each
  message consumes two requests (transcribe + polish). When the limit is hit, the bot tells you how
  long to wait before retrying.
- `test_pipeline.py` runs the transcribe + polish pipeline on `audio.ogg` without Telegram:
  `.venv/Scripts/python.exe test_pipeline.py`.
