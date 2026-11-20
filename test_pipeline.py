"""Manual end-to-end test: transcribe audio.ogg and polish it, without Telegram.

Usage: .venv/Scripts/python.exe test_pipeline.py
"""

import asyncio
import os
from pathlib import Path

from groq import AsyncGroq

from main import polish_transcription, transcribe_audio


async def run() -> None:
    audio = Path(__file__).with_name("audio.ogg")
    data = audio.read_bytes()

    groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    raw = (await transcribe_audio(groq, audio.name, data)).strip()
    print("=== RAW TRANSCRIPTION ===")
    print(raw)
    print()

    polished = await polish_transcription(groq, raw)
    print("=== POLISHED ===")
    print(polished)


if __name__ == "__main__":
    asyncio.run(run())
