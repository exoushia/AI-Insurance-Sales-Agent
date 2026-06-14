"""
voice/run_voice.py — Pipecat WebRTC voice bot for the Insurance Sales Agent
==========================================================================
Wraps the text `ConversationOrchestrator` in a real-time voice pipeline:

    browser mic → WebRTC → Silero VAD → Sarvam Saaras STT
        → OrchestratorProcessor (FSM + 15 sub-agents + RAG)
        → Sarvam Bulbul TTS → WebRTC → browser speaker

Run it, then open the printed URL (http://localhost:7860/client) and click
Connect to talk to the agent. All tunables live in voice/voice_config.py.

    python -m voice.run_voice      (from the AI-Insurance-Sales-Agent dir)
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_DIR, ".env"))

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import TransportParams

from orchestrator import ConversationOrchestrator
from voice.orchestrator_processor import OrchestratorProcessor
from voice.voice_config import VOICE_CONFIG


def _build_stt() -> SarvamSTTService:
    cfg = VOICE_CONFIG.stt
    # `mode` stays a top-level arg (not part of Settings); model/language/VAD go
    # through the non-deprecated Settings API.
    return SarvamSTTService(
        api_key=VOICE_CONFIG.sarvam_api_key,
        mode=cfg.mode,
        settings=SarvamSTTService.Settings(
            model=cfg.model,
            language=cfg.language,
            vad_signals=cfg.vad_signals,
            high_vad_sensitivity=cfg.high_vad_sensitivity,
        ),
    )


def _build_tts() -> SarvamTTSService:
    cfg = VOICE_CONFIG.tts
    # `sample_rate` stays a top-level arg; model/voice/prosody go through Settings.
    return SarvamTTSService(
        api_key=VOICE_CONFIG.sarvam_api_key,
        sample_rate=cfg.sample_rate,
        settings=SarvamTTSService.Settings(
            model=cfg.model,
            voice=cfg.voice,
            language=cfg.language,
            pace=cfg.pace,
            pitch=cfg.pitch,
            loudness=cfg.loudness,
            enable_preprocessing=cfg.enable_preprocessing,
        ),
    )


async def bot(runner_args: RunnerArguments) -> None:
    """Pipecat entry point — built and run once per browser connection."""
    tcfg = VOICE_CONFIG.transport

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=tcfg.audio_in_enabled,
            audio_out_enabled=tcfg.audio_out_enabled,
            audio_out_sample_rate=tcfg.audio_out_sample_rate,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)

    # One orchestrator per connection → one ConversationRecord / session.
    orchestrator = ConversationOrchestrator(
        session_id=VOICE_CONFIG.conversation.session_id
    )

    stt = _build_stt()
    tts = _build_tts()
    brain = OrchestratorProcessor(orchestrator, VOICE_CONFIG)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            brain,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=tcfg.enable_interruptions,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("[voice] client connected — sending greeting")
        await task.queue_frames(
            [TTSSpeakFrame(VOICE_CONFIG.conversation.greeting)]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("[voice] client disconnected — ending session")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
