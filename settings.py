"""
Application configuration for local development.

All model names are centralized here so switching models requires edits in one file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at import time
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# Drop empty base-URL env vars: the OpenAI/Sarvam SDKs read these directly, and an
# empty string ("") is treated as a malformed base URL ("missing protocol"). We
# only want them set when they hold a real URL.
for _empty_base_var in ("OPENAI_BASE_URL", "SARVAM_BASE_URL"):
    if os.environ.get(_empty_base_var, "").strip() == "":
        os.environ.pop(_empty_base_var, None)


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ModelCatalog:
    """Centralized model map used by all subagents.

    Cheap, structured tasks (intent label, schema NER, yes/no validation) run on
    a smaller/faster model; only the customer-facing copy + RAG synthesis
    (`response_generator`) needs the larger model for quality.
    """

    understanding: str = _env_str("OPENAI_MODEL_UNDERSTANDING", "gpt-4.1-mini")
    intent_classifier: str = _env_str("OPENAI_MODEL_INTENT", "gpt-4.1-nano")
    schema_extractor: str = _env_str("OPENAI_MODEL_SCHEMA", "gpt-4.1-nano")
    response_generator: str = _env_str("OPENAI_MODEL_RESPONSE", "gpt-4.1-mini")
    response_validator: str = _env_str("OPENAI_MODEL_VALIDATOR", "gpt-4.1-nano")
    translator: str = _env_str("OPENAI_MODEL_TRANSLATION", "gpt-4.1-nano")


@dataclass(frozen=True)
class AppConfig:
    """Runtime config loaded from environment variables."""

    openai_api_key: str = _env_str("OPENAI_API_KEY", "")
    openai_base_url: str = _env_str("OPENAI_BASE_URL", "")
    sarvam_api_key: str = _env_str("SARVAM_API_KEY", "")
    sarvam_base_url: str = _env_str("SARVAM_BASE_URL", "")
    openai_timeout_seconds: int = _env_int("OPENAI_TIMEOUT_SECONDS", 25)
    openai_temperature: float = _env_float("OPENAI_TEMPERATURE", 0.2)
    fallback_mode: bool = _env_bool("APP_FALLBACK_MODE", True)
    # Which conversation engine to use:
    #   "fsm"     → deterministic finite-state-machine orchestrator (default, safe).
    #   "agentic" → LLM native tool-calling orchestrator (M_16), FSM as fallback.
    orchestration_mode: str = _env_str("ORCHESTRATION_MODE", "fsm")
    models: ModelCatalog = ModelCatalog()

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def sarvam_enabled(self) -> bool:
        return bool(self.sarvam_api_key)

    @property
    def agentic_enabled(self) -> bool:
        return self.orchestration_mode.strip().lower() == "agentic"
