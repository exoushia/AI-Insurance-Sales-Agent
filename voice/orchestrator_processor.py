"""
voice/orchestrator_processor.py
===============================
The bridge between Pipecat's audio frames and the text `ConversationOrchestrator`.

Pipeline position:  ... → SarvamSTT → [OrchestratorProcessor] → SarvamTTS → ...

On each FINAL transcription it:
  1. runs `orchestrator.process_message(text)` in a thread executor (the
     orchestrator makes BLOCKING OpenAI/Sarvam calls — must not block the loop),
  2. switches the TTS voice to the user's detected language for this turn,
  3. speaks the assistant's reply.

All other frames pass straight through.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .voice_config import VoiceConfig, tts_language_for


class OrchestratorProcessor(FrameProcessor):
    """Routes user transcripts through the orchestrator and speaks the reply."""

    def __init__(self, orchestrator, config: VoiceConfig):
        super().__init__()
        self._orchestrator = orchestrator
        self._config = config

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Only act on final user transcripts; everything else flows through.
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                await self._handle_user_text(text)
            return

        await self.push_frame(frame, direction)

    async def _handle_user_text(self, text: str) -> None:
        logger.info(f"[voice] user: {text}")

        # The orchestrator turn is fully synchronous and network-bound; run it
        # off the event loop so audio I/O keeps flowing.
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            result = await loop.run_in_executor(
                None, self._orchestrator.process_message, text
            )
        except Exception as exc:  # never kill the call on a turn failure
            logger.exception(f"[voice] orchestrator error: {exc}")
            await self.push_frame(
                TTSSpeakFrame("Sorry, I ran into a problem. Could you say that again?")
            )
            return

        elapsed_ms = (loop.time() - started) * 1000.0
        reply = result.get("assistant_text", "") or ""
        schema_language = self._orchestrator.record.schema.language
        logger.info(
            f"[voice] assistant ({result.get('state')}, lang={schema_language}, "
            f"{elapsed_ms:.0f}ms): {reply}"
        )

        # Match the TTS voice to the language the reply was rendered in.
        tts_language = tts_language_for(schema_language)
        await self.push_frame(
            TTSUpdateSettingsFrame(settings={"language": tts_language})
        )

        if reply:
            await self.push_frame(TTSSpeakFrame(reply))
