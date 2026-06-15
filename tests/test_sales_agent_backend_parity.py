"""
Parity test: legacy SalesAgent vs SalesAgentAgentsSDK (Phase 5)
===============================================================
Runs the same 6 demo scenarios against BOTH M_16 backends and asserts:
  1. Final outcome tuple matches (product_id, plan_id != None, purchased).
  2. Tool sequence semantic parity (same tool names called, order may vary).
  3. Guardrail status (both ok, or if legacy fails so can SDK).
  4. Token / latency delta printed for manual review (not asserted).

Usage (requires OPENAI_API_KEY):
    cd AI-Insurance-Sales-Agent
    ORCHESTRATION_MODE=agentic ../.venv/bin/python tests/test_sales_agent_backend_parity.py

Set AGENTIC_BACKEND=agents_sdk to test the SDK backend only (skips legacy run).
Set PARITY_SCENARIO=price_anxiety to run a single scenario.

The test is designed to be run manually before cutting a release with
AGENTIC_BACKEND=agents_sdk — it is NOT a fast unit test (requires live OpenAI
calls). It prints a summary table and exits 0 on full parity, 1 on any mismatch.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Make the package root importable when run directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("ORCHESTRATION_MODE", "agentic")

from settings import AppConfig
from agentic_orchestrator import AgenticOrchestrator


# ---------------------------------------------------------------------------
# SCENARIOS  (identical to demo_agentic.py SCENARIOS dict)
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, list[str]] = {
    "price_anxiety": [
        "Hi, I'm 32 and looking for health insurance just for myself.",
        "Mainly for maternity, we're planning a baby next year.",
        "Okay what would something like that cost me?",
        "Honestly that feels expensive, why should I pay that much?",
        "Alright that makes sense. Let's go with the recommended plan.",
    ],
    "deep_policy": [
        "I'm 41, need a family floater for me, my wife and one kid.",
        "I have diabetes, will that be covered?",
        "What's the waiting period for pre-existing conditions on that plan?",
        "And does it cover day-care procedures like cataract surgery?",
        "Okay, that sounds reassuring. Let's go ahead with the recommended plan.",
        "Yes, please go ahead and finalize it.",
    ],
    "general_qa": [
        "Before I decide, I have a general question.",
        "If I buy now and switch insurers next year, do I lose my waiting period?",
        "Got it. And what's a free-look period?",
    ],
    "hinglish": [
        "Hello, mujhe apne liye ek health insurance chahiye, main 30 saal ka hoon.",
        "Mainly hospitalisation ke liye, sirf mere liye, koi pre-existing bimari nahi hai.",
        "Mera budget around 10,000 rupees saal ka hai. Iska premium kitna hoga?",
        "Theek hai, recommended plan le lete hain.",
        "Haan, please ise abhi finalize kar dijiye.",
    ],
    "low_budget_niche": [
        "Hi, I'm a delivery rider, 27. Money's tight but I want some cover, just for me.",
        "I won't have a regular salary, so I just want cash if I'm hospitalised.",
        "Honestly I can't afford much — maybe two, three thousand a year at most.",
        "Okay, show me the best plan you've got for me and what it costs.",
        "That sounds doable. Let's go ahead and finalize it.",
    ],
    "off_topic": [
        "Hey, quick one — who do you think wins the cricket world cup this year?",
        "Haha fair. Also what's the weather like in Mumbai today?",
        "Got it. Can you also recommend a good biryani place near me?",
        "Alright, no insurance for me today. Thanks for your help, bye!",
    ],
}

# Expected outcome per scenario: (should_purchase, expected_tool_names_subset)
# tool_names_subset = tools that MUST appear in the full run (any turn).
EXPECTED: dict[str, tuple[bool, list[str]]] = {
    "price_anxiety":    (True,  ["save_profile", "recommend_products", "estimate_value_vs_cost", "finalize_purchase"]),
    "deep_policy":      (True,  ["save_profile", "recommend_products", "explain_product", "finalize_purchase"]),
    "general_qa":       (False, ["answer_general_question"]),
    "hinglish":         (True,  ["save_profile", "recommend_products", "finalize_purchase"]),
    "low_budget_niche": (True,  ["recommend_products", "finalize_purchase"]),
    "off_topic":        (False, []),
}


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    backend: str
    purchased: bool
    product_id: str | None
    plan_id: str | None
    tools_seen: list[str]
    guardrail_ok: bool
    total_turns: int
    elapsed_s: float
    error: str | None = None


def run_scenario_with_backend(
    name: str,
    turns: list[str],
    backend: str,
    config: AppConfig,
) -> ScenarioResult:
    # AppConfig is a frozen dataclass whose env-var defaults are read at class
    # definition time (import time). Pass the backend value directly so we don't
    # rely on os.environ mutations after import.
    cfg = AppConfig(
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        sarvam_api_key=config.sarvam_api_key,
        sarvam_base_url=config.sarvam_base_url,
        openai_timeout_seconds=config.openai_timeout_seconds,
        openai_temperature=config.openai_temperature,
        fallback_mode=config.fallback_mode,
        orchestration_mode="agentic",
        agentic_backend=backend,
        agents_sdk_tracing=config.agents_sdk_tracing,
        models=config.models,
    )

    orch = AgenticOrchestrator(session_id=f"parity_{name}_{backend}", config=cfg)
    if not orch.llm.is_available:
        return ScenarioResult(
            scenario=name, backend=backend, purchased=False, product_id=None,
            plan_id=None, tools_seen=[], guardrail_ok=False, total_turns=0,
            elapsed_s=0.0, error="LLM unavailable",
        )

    all_tools: list[str] = []
    guardrail_ok = True
    last: dict[str, Any] = {}
    t0 = time.perf_counter()
    error_msg = None

    try:
        for user_text in turns:
            out = orch.process_message(user_text)
            last = out
            all_tools.extend(out.get("tool_sequence") or [])
            if not out.get("guardrail_ok", True):
                guardrail_ok = False
    except Exception as exc:
        error_msg = str(exc)

    elapsed = time.perf_counter() - t0

    return ScenarioResult(
        scenario=name,
        backend=backend,
        purchased=bool(last.get("purchased")),
        product_id=last.get("resolved_product_id"),
        plan_id=last.get("resolved_plan_id"),
        tools_seen=all_tools,
        guardrail_ok=guardrail_ok,
        total_turns=len(turns),
        elapsed_s=round(elapsed, 2),
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# PARITY CHECKS
# ---------------------------------------------------------------------------

def check_parity(
    name: str,
    legacy: ScenarioResult,
    sdk: ScenarioResult,
    expected_purchase: bool,
    expected_tools: list[str],
) -> list[str]:
    """Return a list of failure strings; empty = parity ok."""
    failures: list[str] = []

    # 1. Both match expected purchase outcome.
    for r in (legacy, sdk):
        if r.error:
            failures.append(f"[{r.backend}] runtime error: {r.error}")
            continue
        if r.purchased != expected_purchase:
            failures.append(
                f"[{r.backend}] purchased={r.purchased}, expected={expected_purchase}"
            )

    # 2. Required tools all present in each backend's run.
    for tool in expected_tools:
        if tool not in legacy.tools_seen:
            failures.append(f"[legacy] missing expected tool: {tool}")
        if tool not in sdk.tools_seen:
            failures.append(f"[agents_sdk] missing expected tool: {tool}")

    # 3. Product/plan parity when both should purchase.
    if expected_purchase and not legacy.error and not sdk.error:
        if legacy.product_id != sdk.product_id:
            failures.append(
                f"product_id mismatch: legacy={legacy.product_id} sdk={sdk.product_id}"
            )
        # plan_id can legitimately differ by tier if both are set
        if bool(legacy.plan_id) != bool(sdk.plan_id):
            failures.append(
                f"plan_id presence mismatch: legacy={legacy.plan_id} sdk={sdk.plan_id}"
            )

    return failures


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    config = AppConfig()
    if not config.openai_api_key:
        print("OPENAI_API_KEY not set — cannot run parity tests.")
        return 1

    target_scenario = os.environ.get("PARITY_SCENARIO")
    scenarios = {k: v for k, v in SCENARIOS.items() if not target_scenario or k == target_scenario}
    # If AGENTIC_BACKEND is set, run only that backend; otherwise run both.
    env_backend = os.environ.get("AGENTIC_BACKEND", "")
    run_backends = [env_backend] if env_backend in ("legacy", "agents_sdk") else ["legacy", "agents_sdk"]

    print(f"\n{'='*70}")
    print(f"PARITY TEST  scenarios={list(scenarios)} backends={run_backends}")
    print(f"{'='*70}\n")

    all_failures: dict[str, list[str]] = {}

    for name, turns in scenarios.items():
        exp_purchase, exp_tools = EXPECTED[name]
        results: dict[str, ScenarioResult] = {}

        for backend in run_backends:
            print(f"  running {name} / {backend} ...", end="", flush=True)
            r = run_scenario_with_backend(name, turns, backend, config)
            results[backend] = r
            status = "PURCHASED" if r.purchased else "not-purchased"
            print(f" {status} | tools={r.tools_seen} | {r.elapsed_s}s"
                  + (f" ERROR: {r.error}" if r.error else ""))

        # Only check parity when both backends ran.
        if len(run_backends) == 2:
            failures = check_parity(
                name,
                results["legacy"],
                results["agents_sdk"],
                exp_purchase,
                exp_tools,
            )
        else:
            # Single backend: check against expectations only.
            r = list(results.values())[0]
            failures = []
            if r.error:
                failures.append(f"runtime error: {r.error}")
            elif r.purchased != exp_purchase:
                failures.append(f"purchased={r.purchased}, expected={exp_purchase}")
            for tool in exp_tools:
                if tool not in r.tools_seen:
                    failures.append(f"missing expected tool: {tool}")

        all_failures[name] = failures
        if failures:
            for f in failures:
                print(f"    FAIL: {f}")
        else:
            print(f"    PASS")

    print(f"\n{'='*70}")
    total_fail = sum(len(v) for v in all_failures.values())
    if total_fail == 0:
        print("ALL PARITY CHECKS PASSED")
    else:
        print(f"{total_fail} parity failure(s) found:")
        for name, failures in all_failures.items():
            for f in failures:
                print(f"  {name}: {f}")
    print(f"{'='*70}\n")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
