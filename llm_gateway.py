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

_openai_event_logger = logging.getLogger("openai_events")
if not _openai_event_logger.handlers:
    _openai_event_logger.setLevel(logging.INFO)
    _event_handler = logging.FileHandler(_LOG_DIR / "openai_events.jsonl", encoding="utf-8")
    _event_handler.setFormatter(logging.Formatter("%(message)s"))
    _openai_event_logger.addHandler(_event_handler)
    _openai_event_logger.propagate = False


def log_llm_call(agent_id: str, path: str, model: str, ok: bool, detail: str = "") -> None:
    """Record one LLM/fallback decision. `path` is "llm" or "fallback"."""
    _llm_logger.info(
        "%s path=%s model=%s %s %s",
        agent_id, path, model, "ok" if ok else "fail", detail,
    )


def _model_dump(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return None
    return None


def _openai_usage(completion: Any) -> dict[str, Any]:
    usage = _model_dump(getattr(completion, "usage", None)) or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens_details": usage.get("prompt_tokens_details"),
        "completion_tokens_details": usage.get("completion_tokens_details"),
    }


def log_openai_event(
    *,
    operation: str,
    model: str,
    ok: bool,
    latency_ms: float,
    completion: Any | None = None,
    hop: int | None = None,
    tool_names: list[str] | None = None,
    error_type: str | None = None,
    finish_reason: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Append one structured OpenAI event line for observability.

    The payload uses only OpenAI-native response metadata plus local timing.
    """
    event = {
        "ts": int(time.time() * 1000),
        "operation": operation,
        "ok": bool(ok),
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "request_id": getattr(completion, "_request_id", None),
        "openai_response_id": getattr(completion, "id", None),
        "created": getattr(completion, "created", None),
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "usage": _openai_usage(completion) if completion is not None else None,
        "finish_reason": finish_reason,
        "hop": hop,
        "tool_names": tool_names or [],
        "error_type": error_type,
        "context": context or {},
    }
    _openai_event_logger.info(json.dumps(event, ensure_ascii=False, default=str))


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

    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        *,
        agent_id: str = "unknown",
        obs_context: dict[str, Any] | None = None,
    ) -> str | None:
        """Return assistant text or None if generation is unavailable/failed."""
        if not self.is_available:
            return None

        def _call():
            return self._client.chat.completions.create(
                model=self.config.models.response_generator,
                temperature=self.config.openai_temperature,
                timeout=self.config.openai_timeout_seconds,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

        started = time.perf_counter()
        completion = self._with_retries(_call)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if completion is None:
            log_openai_event(
                operation="chat.completions.generate_response",
                model=self.config.models.response_generator,
                ok=False,
                latency_ms=latency_ms,
                error_type="api_fail_or_retry_exhausted",
                context={"agent_id": agent_id, **(obs_context or {})},
            )
            return None
        choice = completion.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        log_openai_event(
            operation="chat.completions.generate_response",
            model=self.config.models.response_generator,
            ok=True,
            latency_ms=latency_ms,
            completion=completion,
            finish_reason=finish_reason,
            context={"agent_id": agent_id, **(obs_context or {})},
        )
        return content.strip() if isinstance(content, str) else None

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
        agent_id: str = "unknown",
        obs_context: dict[str, Any] | None = None,
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
            return self._client.chat.completions.create(**kwargs)

        for use_format in (True, False):   # retry without response_format if rejected
            started = time.perf_counter()
            completion = self._with_retries(lambda: _call(use_format))
            latency_ms = (time.perf_counter() - started) * 1000.0
            if completion is None:
                log_openai_event(
                    operation="chat.completions.complete_json",
                    model=use_model,
                    ok=False,
                    latency_ms=latency_ms,
                    error_type="api_fail_or_retry_exhausted",
                    context={
                        "agent_id": agent_id,
                        "response_format": "json_object" if use_format else "none",
                        **(obs_context or {}),
                    },
                )
                continue

            choice = completion.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)
            if isinstance(content, str):
                try:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    parsed = json.loads(match.group(0) if match else content)
                    log_openai_event(
                        operation="chat.completions.complete_json",
                        model=use_model,
                        ok=True,
                        latency_ms=latency_ms,
                        completion=completion,
                        finish_reason=finish_reason,
                        context={
                            "agent_id": agent_id,
                            "response_format": "json_object" if use_format else "none",
                            **(obs_context or {}),
                        },
                    )
                    return parsed
                except Exception:
                    pass
            log_openai_event(
                operation="chat.completions.complete_json",
                model=use_model,
                ok=False,
                latency_ms=latency_ms,
                completion=completion,
                finish_reason=finish_reason,
                error_type="json_parse_failed",
                context={
                    "agent_id": agent_id,
                    "response_format": "json_object" if use_format else "none",
                    **(obs_context or {}),
                },
            )
        return None

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_hops: int = 4,
        agent_id: str = "M_16",
        obs_context: dict[str, Any] | None = None,
    ) -> dict | None:
        """
        Native OpenAI tool-calling loop. This is the engine of the agentic
        orchestrator: the model decides which tools to call, we execute them via
        `dispatch(name, args) -> dict`, feed results back, and repeat until the
        model returns a plain text reply (or we hit `max_hops`).

        `messages` is mutated in place so the caller keeps the full trace (system,
        user, assistant tool_calls, tool results) for observability/persistence.

        Returns a dict:
            {
              "text":        final assistant text (str),
              "tool_trace":  [ {name, arguments, result}, ... ],
              "hops":        number of tool-calling rounds,
            }
        or None if the LLM is unavailable / all attempts failed.
        """
        if not self.is_available:
            return None

        use_model = model or self.config.models.response_generator
        temp = self.config.openai_temperature if temperature is None else temperature
        tool_trace: list[dict] = []

        for hop in range(max_hops + 1):
            def _call():
                kwargs: dict[str, Any] = {
                    "model": use_model,
                    "temperature": temp,
                    "timeout": self.config.openai_timeout_seconds,
                    "messages": messages,
                }
                # On the final allowed hop, force a text answer (no more tools).
                if hop < max_hops:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return self._client.chat.completions.create(**kwargs)

            started = time.perf_counter()
            completion = self._with_retries(_call)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if completion is None:
                log_openai_event(
                    operation="chat.completions.generate_with_tools",
                    model=use_model,
                    ok=False,
                    latency_ms=latency_ms,
                    hop=hop,
                    error_type="api_fail_or_retry_exhausted",
                    context={"agent_id": agent_id, **(obs_context or {})},
                )
                log_llm_call(agent_id, "fallback", use_model, False, f"hop={hop} api_fail")
                return None

            choice = completion.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)
            finish_reason = getattr(choice, "finish_reason", None)
            tool_names = [tc.function.name for tc in tool_calls] if tool_calls else []
            log_openai_event(
                operation="chat.completions.generate_with_tools",
                model=use_model,
                ok=True,
                latency_ms=latency_ms,
                completion=completion,
                hop=hop,
                tool_names=tool_names,
                finish_reason=finish_reason,
                context={"agent_id": agent_id, **(obs_context or {})},
            )

            if not tool_calls:
                text = msg.content.strip() if isinstance(msg.content, str) else ""
                log_llm_call(agent_id, "llm", use_model, True,
                             f"hops={hop} tools={len(tool_trace)}")
                return {"text": text, "tool_trace": tool_trace, "hops": hop}

            # Record the assistant turn that requested the tools.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each requested tool and feed the result back.
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = dispatch(name, args)
                except Exception as exc:  # pragma: no cover - defensive
                    result = {"error": f"tool '{name}' failed: {exc}"}
                tool_trace.append({"name": name, "arguments": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:8000],
                })

        # Exhausted hops without a final text answer.
        log_llm_call(agent_id, "fallback", use_model, False, "max_hops_no_text")
        return {"text": "", "tool_trace": tool_trace, "hops": max_hops}


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
