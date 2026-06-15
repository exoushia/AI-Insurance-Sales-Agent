"""
voice/voice_config.py — single source of truth for ALL voice knobs
==================================================================
Every Pipecat / Sarvam parameter for the voice layer lives here so the demo can
be tuned in one place. Nothing else in voice/ hard-codes a model name, voice,
language, sample rate, or threshold.

STT  = Sarvam Saaras  (saaras:v3)   — speech → text, language-preserving.
TTS  = Sarvam Bulbul  (bulbul:v3)   — text → speech, voice "shreya".
Transport = SmallWebRTCTransport    — browser mic/speaker at localhost:7860.
VAD  = Silero (Pipecat local)       — turn-taking / barge-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pipecat.transcriptions.language import Language


# ── Speech-to-Text (Sarvam Saaras) ─────────────────────────────────────────
@dataclass
class STTConfig:
    model: str = "saaras:v3"
    # saaras:v3 modes: transcribe | translate | verbatim | translit | codemix.
    # "transcribe" keeps the user's original language + numbers + code-mixing,
    # which is what our multilingual LLM + M_10 output pipeline expects.
    mode: str = "transcribe"
    # Default recognition language. With Silero (local) VAD we leave server-side
    # VAD off; Saaras still auto-handles Hindi/English/Hinglish within a turn.
    language: Language = Language.EN_IN
    vad_signals: bool = False          # rely on Pipecat's local Silero VAD
    high_vad_sensitivity: bool = False


# ── Text-to-Speech (Sarvam Bulbul) ─────────────────────────────────────────
@dataclass
class TTSConfig:
    model: str = "bulbul:v3"           # v3 → temperature control, 24000 Hz
    voice: str = "shreya"              # locked demo voice (v3 female speaker)
    language: Language = Language.EN_IN
    sample_rate: int = 24000           # bulbul:v3 native rate
    # bulbul:v3 ranges: pace 0.5–2.0, temperature 0.01–1.0. v3 does NOT support
    # pitch/loudness (those were v2-only). Lower temperature = steadier, more
    # consistent delivery, which suits a trustworthy insurance agent.
    pace: float = 0.9
    temperature: float = 0.5
    enable_preprocessing: bool = True  # always-on for v3; smooths numbers / mixed text


# ── Voice Activity Detection (Silero, local) ───────────────────────────────
@dataclass
class VADConfig:
    """Turn-taking thresholds for the local Silero VAD.

    stop_secs is the critical anti-fragmentation knob: it is how long the user
    must stay silent before the VAD declares the turn finished. Pipecat's
    default (0.2s) is far too eager for spontaneous speech — a caller who pauses
    mid-sentence to think ("I want maternity cover... and, um, dental too") gets
    chopped into several tiny transcripts, each driving its own orchestrator
    turn. The result is a flood of half-answered questions that sounds like the
    agent is looping. 0.8s lets a normal speaker finish a thought before we
    treat the turn as over, merging paused phrases into one coherent utterance.
    Barge-in still works (start_secs is short), so the user can interrupt.
    """
    confidence: float = 0.7   # Silero speech-probability threshold (default)
    start_secs: float = 0.2   # speech must persist this long to start a turn
    stop_secs: float = 0.8    # silence this long ends the turn (was 0.2 default)
    min_volume: float = 0.6   # ignore audio quieter than this (default)


# ── Transport (SmallWebRTC, browser) ───────────────────────────────────────
@dataclass
class TransportConfig:
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    enable_interruptions: bool = True  # barge-in: user can cut in mid-reply
    audio_out_sample_rate: int = 24000  # match TTSConfig.sample_rate (v3)


# ── Conversation / demo text ───────────────────────────────────────────────
@dataclass
class ConversationConfig:
    session_id: str = "voice_demo"
    # Spoken when the browser client connects (S0 bootstrap; the FSM takes over
    # on the user's first reply). Kept English; switches per-turn after that.
    greeting: str = (
        "Hello! I'm your Swasthya health insurance assistant. "
        "How can I help you find the right cover today?"
    )


# ── Language mapping (our schema label → Pipecat TTS Language) ──────────────
# fsm.detect_input_language returns "english" | "hindi" | "hinglish"; schema may
# also be None (untouched = English). Hindi + Hinglish both speak via hi-IN
# (M_10 already renders Hinglish in roman/code-mix that Bulbul reads in Hindi).
_LANGUAGE_TO_TTS: dict[str, Language] = {
    "english": Language.EN_IN,
    "hindi": Language.HI,
    "hinglish": Language.HI,
}


def tts_language_for(schema_language: str | None) -> Language:
    """Map a schema language label to the Bulbul TTS Language enum."""
    if not schema_language:
        return Language.EN_IN
    return _LANGUAGE_TO_TTS.get(schema_language.lower(), Language.EN_IN)


# ── Top-level bundle ───────────────────────────────────────────────────────
@dataclass
class VoiceConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    @property
    def sarvam_api_key(self) -> str:
        return os.getenv("SARVAM_API_KEY", "")


# One shared instance the rest of voice/ imports.
VOICE_CONFIG = VoiceConfig()
