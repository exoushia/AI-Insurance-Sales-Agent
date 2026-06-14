"""
OpenAI SDK gateway with safe fallback behavior.

This wrapper isolates SDK usage and keeps orchestrator logic simple.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from settings import AppConfig

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at import time
    OpenAI = None

# Transient OpenAI failures (rate limit, timeout, brief connection blips) are
# worth a quick retry; on a fast voice loop a swallowed one silently degrades a
# turn to the deterministic fallback (e.g. intent → "unrecognised").
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
    _TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
        APIConnectionError, APITimeoutError, RateLimitError, InternalServerError,
    )
except Exception:  # pragma: no cover - SDK shape changed / not installed
    _TRANSIENT_ERRORS = ()

try:
    from sarvamai import SarvamAI
except Exception:  # pragma: no cover - optional dependency at import time
    SarvamAI = None


# ---------------------------------------------------------------------------
# LLM CALL LOGGER  (append-only trail of which path each agent took)
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_llm_logger = logging.getLogger("llm_calls")
if not _llm_logger.handlers:
    _llm_logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(_LOG_DIR / "llm_calls.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _llm_logger.addHandler(_handler)
    _llm_logger.propagate = False


def log_llm_call(agent_id: str, path: str, model: str, ok: bool, detail: str = "") -> None:
    """Record one LLM/fallback decision. `path` is "llm" or "fallback"."""
    _llm_logger.info(
        "%s path=%s model=%s %s %s",
        agent_id, path, model, "ok" if ok else "fail", detail,
    )


class LLMGateway:
    """Thin OpenAI chat wrapper used by text-mode orchestration."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
        self._init_error = None

        if not config.openai_enabled:
            self._init_error = "OPENAI_API_KEY missing"
            return

        if OpenAI is None:
            self._init_error = "openai package not installed"
            return

        kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        elif os.environ.get("OPENAI_BASE_URL") == "":
            # An EMPTY `OPENAI_BASE_URL` in the environment poisons the OpenAI SDK:
            # it reads the env var directly and hands httpx a URL with no scheme
            # ("Request URL is missing an 'http://' or 'https://' protocol"), so
            # every call fails with APIConnectionError. Drop the empty value so the
            # SDK falls back to its official endpoint.
            os.environ.pop("OPENAI_BASE_URL", None)

        self._client = OpenAI(**kwargs)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def generate_response(self, system_prompt: str, user_message: str) -> str | None:
        """Return assistant text or None if generation is unavailable/failed."""
        if not self.is_available:
            return None

        def _call():
            completion = self._client.chat.completions.create(
                model=self.config.models.response_generator,
                temperature=self.config.openai_temperature,
                timeout=self.config.openai_timeout_seconds,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            content = completion.choices[0].message.content
            return content.strip() if isinstance(content, str) else None

        return self._with_retries(_call)

    def _with_retries(self, call, attempts: int = 2):
        """Run `call`, retrying transient OpenAI errors once with a short backoff.
        Kept deliberately light so a voice turn rides out a single blip without
        stalling; returns None if all attempts fail or on any non-transient error."""
        for attempt in range(attempts):
            try:
                return call()
            except _TRANSIENT_ERRORS:
                if attempt == attempts - 1:
                    return None
                time.sleep(0.3)
            except Exception:
                return None
        return None

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict | None:
        """Return a parsed JSON object from the model, or None on failure."""
        if not self.is_available:
            return None
        use_model = model or self.config.models.intent_classifier
        temp = self.config.openai_temperature if temperature is None else temperature

        def _call(use_format: bool):
            kwargs: dict[str, Any] = {
                "model": use_model,
                "temperature": temp,
                "timeout": self.config.openai_timeout_seconds,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }
            if use_format:
                kwargs["response_format"] = {"type": "json_object"}
            completion = self._client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if not isinstance(content, str):
                return None
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(match.group(0) if match else content)

        for use_format in (True, False):   # retry without response_format if rejected
            result = self._with_retries(lambda: _call(use_format))
            if result is not None:
                return result
        return None


class SarvamGateway:
    """Thin Sarvam translation wrapper used by M_10 for Hindi/Hinglish output."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
        self._init_error = None

        if not config.sarvam_enabled:
            self._init_error = "SARVAM_API_KEY missing"
            return
        if SarvamAI is None:
            self._init_error = "sarvamai package not installed"
            return
        try:
            self._client = SarvamAI(api_subscription_key=config.sarvam_api_key)
        except Exception as exc:  # pragma: no cover - defensive
            self._init_error = f"sarvam init failed: {exc}"

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def translate(
        self,
        text: str,
        *,
        target_language_code: str,
        mode: str = "formal",
        output_script: str = "fully-native",
        source_language_code: str = "en-IN",
    ) -> str | None:
        """Translate English text. Returns translated text or None on failure.

        Numerals are kept in international (digit) form so the numeric guardrail
        can re-validate amounts after translation.
        """
        if not self.is_available or not text.strip():
            return None
        try:
            resp = self._client.text.translate(
                input=text,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                mode=mode,
                output_script=output_script,
                numerals_format="international",
            )
            translated = getattr(resp, "translated_text", None)
            return translated.strip() if isinstance(translated, str) else None
        except Exception:
            return None
