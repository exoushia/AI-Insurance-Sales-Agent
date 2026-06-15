"""
tools/session_analytics.py — Post-session trace analyser
=========================================================
Reads the two log files produced every run and renders three charts:

  Chart 1 — Tokens per sub-agent
      Stacked bar (prompt + completion) from openai_events.jsonl.
      Quickly shows which agents are the costliest callers.

  Chart 2 — Latency per sub-agent
      Per-LLM-call latency (ms) from openai_events.jsonl, and
      per-turn wall-clock latency from the conversation log.
      Reveals where the pipeline is slow.

  Chart 3 — Intent × Tool-call timeline
      One row per conversation turn.  Left cell = user intent (colour coded).
      Right cell = tools the agent called (or agents fired for FSM sessions).
      The most important chart for understanding the call story.

Usage:
    # latest session in logs/
    ../.venv/bin/python tools/session_analytics.py

    # specific session
    ../.venv/bin/python tools/session_analytics.py --session demo_price_anxiety

    # open the PNG after saving
    ../.venv/bin/python tools/session_analytics.py --show

Output: logs/analytics_<session_id>.png  (and optionally displayed in a window)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # headless by default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_LOGS = _ROOT / "logs"

# ── colour palette ─────────────────────────────────────────────────────────
INTENT_COLORS = {
    "provide_info":       "#4C9BE8",
    "prospective":        "#27AE60",
    "inquiry":            "#F39C12",
    "exploratory":        "#9B59B6",
    "ask_policy_question":"#1ABC9C",
    "explore_more":       "#E67E22",
    "want_human":         "#E74C3C",
    "frustrated":         "#C0392B",
    "done":               "#95A5A6",
    "unsafe":             "#922B21",
    "unrecognised":       "#BDC3C7",
}
TOOL_COLORS = {
    "save_profile":           "#3498DB",
    "recommend_products":     "#2ECC71",
    "explain_product":        "#9B59B6",
    "show_plan_options":      "#1ABC9C",
    "estimate_value_vs_cost": "#F39C12",
    "answer_general_question":"#E67E22",
    "finalize_purchase":      "#E74C3C",
}
_DEFAULT_TOOL_COLOR = "#7F8C8D"
_PROMPT_COLOR  = "#5B9BD5"
_COMP_COLOR    = "#ED7D31"
_LATENCY_COLOR = "#70AD47"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _latest_session_id() -> str:
    """Return the session_id of the most recently modified conversation log."""
    conv_logs = sorted(_LOGS.glob("conversation_*.json"), key=lambda p: p.stat().st_mtime)
    if not conv_logs:
        sys.exit("No conversation_*.json found in logs/")
    data = json.loads(conv_logs[-1].read_text())
    return data.get("session_id") or conv_logs[-1].stem.replace("conversation_", "")


def load_conversation(session_id: str) -> dict[str, Any]:
    path = _LOGS / f"conversation_{session_id}.json"
    if not path.exists():
        # try glob in case session_id contains special chars
        candidates = list(_LOGS.glob(f"conversation_{session_id}*.json"))
        if not candidates:
            sys.exit(f"No conversation log for session '{session_id}'")
        path = candidates[0]
    data = json.loads(path.read_text())
    return data  # {session_id, language, started_at, turns: [...]}


def load_openai_events(session_id: str) -> list[dict]:
    events_path = _LOGS / "openai_events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctx = e.get("context", {})
        if ctx.get("session_id") == session_id:
            events.append(e)
    return events


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_tokens(events: list[dict]) -> dict[str, dict[str, int]]:
    """Return {agent_id: {prompt: N, completion: N}} from openai events."""
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0})
    for e in events:
        if not e.get("ok"):
            continue
        agent = e.get("context", {}).get("agent_id", "unknown")
        usage = e.get("usage") or {}
        totals[agent]["prompt"]     += usage.get("prompt_tokens", 0)
        totals[agent]["completion"] += usage.get("completion_tokens", 0)
    return dict(totals)


def aggregate_latency(events: list[dict]) -> dict[str, list[float]]:
    """Return {agent_id: [latency_ms, ...]} per LLM call."""
    data: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if not e.get("ok"):
            continue
        agent = e.get("context", {}).get("agent_id", "unknown")
        if e.get("latency_ms") is not None:
            data[agent].append(float(e["latency_ms"]))
    return dict(data)


def turn_wall_latencies(turns: list[dict]) -> list[float]:
    """Inter-turn wall-clock gaps (seconds) derived from logged_at timestamps."""
    result: list[float] = []
    prev_dt = None
    for t in turns:
        ts_str = t.get("logged_at", "")
        try:
            dt = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            result.append(0.0)
            continue
        if prev_dt is None:
            result.append(0.0)
        else:
            result.append((dt - prev_dt).total_seconds())
        prev_dt = dt
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Agent invocation frequency (always) + token overlay (agentic only)
# ═══════════════════════════════════════════════════════════════════════════

# Agent category colours for the frequency bars
_AGENT_CATEGORY_COLOR: dict[str, str] = {
    "M_01": "#4C9BE8",   # intent
    "M_02": "#E74C3C",   # escalation
    "M_03": "#27AE60",   # closure
    "M_04": "#F39C12",   # schema extractor
    "M_05": "#9B59B6",   # probing
    "M_06": "#1ABC9C",   # policy retrieval
    "M_07": "#2980B9",   # policy summary
    "M_08": "#E67E22",   # policy QA
    "M_09": "#16A085",   # RAG
    "M_10": "#8E44AD",   # translator
    "M_11": "#2C3E50",   # response queue
    "M_12": "#7F8C8D",   # validator
    "M_13": "#BDC3C7",   # analytics
    "M_14": "#27AE60",   # whatsapp
    "M_15": "#D35400",   # guardrail
    "M_16": "#C0392B",   # sales agent
    "M_16_agents_sdk": "#922B21",
}

def _agent_invocation_counts(turns: list[dict]) -> dict[str, int]:
    """Count how many turns each agent was fired in (conv log agents_fired)."""
    counts: dict[str, int] = defaultdict(int)
    for t in turns:
        for a in (t.get("agents_fired") or []):
            counts[str(a)] = counts[str(a)] + 1
    return dict(counts)


def _chart_agent_frequency(
    ax: plt.Axes,
    turns: list[dict],
    token_data: dict[str, dict[str, int]],
) -> None:
    counts = _agent_invocation_counts(turns)
    if not counts:
        ax.text(0.5, 0.5, "No agents_fired data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#666")
        ax.set_title("Chart 1 — Agent Invocation Frequency", fontweight="bold")
        return

    agents = sorted(counts.keys(), key=lambda a: -counts[a])
    freqs  = [counts[a] for a in agents]
    colors = [_AGENT_CATEGORY_COLOR.get(a, "#95A5A6") for a in agents]
    x = range(len(agents))

    bars = ax.bar(x, freqs, color=colors, zorder=2, edgecolor="white", linewidth=0.5)

    for xi, a, f in zip(x, agents, freqs):
        label = str(f)
        # Append total token count if available (agentic sessions)
        if a in token_data:
            tok = token_data[a]["prompt"] + token_data[a]["completion"]
            label = f"{f}\n({tok:,} tok)"
        ax.text(xi, f + 0.05, label, ha="center", va="bottom",
                fontsize=7.5, color="#333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(agents, rotation=0, fontsize=9)
    ax.set_ylabel("Turns fired in", fontsize=10)
    ax.set_title("Chart 1 — Agent Invocation Frequency\n"
                 "(number shows turns active; token count where available)",
                 fontweight="bold", pad=6, fontsize=10)
    ax.set_ylim(0, max(freqs) * 1.35)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Latency (2a) + Turn complexity (2b)
# ═══════════════════════════════════════════════════════════════════════════

def _chart_latency(ax_agent: plt.Axes, ax_turn: plt.Axes,
                   latency_data: dict[str, list[float]],
                   turns: list[dict],
                   token_data: dict[str, dict[str, int]]) -> None:
    # ── 2a: per-agent LLM latency (agentic only) ──────────────────────────
    if latency_data:
        agents  = list(latency_data.keys())
        avg_lat = [sum(v) / len(v) for v in latency_data.values()]
        total_calls = [len(v) for v in latency_data.values()]
        x = range(len(agents))
        ax_agent.bar(x, avg_lat, color=_LATENCY_COLOR, zorder=2,
                     edgecolor="white", linewidth=0.5)
        for xi, avg, n in zip(x, avg_lat, total_calls):
            ax_agent.text(xi, avg + max(avg_lat) * 0.02,
                          f"{avg:.0f} ms\n({n} calls)",
                          ha="center", va="bottom", fontsize=8, color="#333")
        ax_agent.set_xticks(list(x))
        ax_agent.set_xticklabels(agents, rotation=0, fontsize=10)
        ax_agent.set_ylabel("Avg LLM latency (ms)", fontsize=10)
        ax_agent.set_ylim(0, max(avg_lat) * 1.4)
    else:
        ax_agent.text(0.5, 0.5,
                      "No LLM latency data\n(FSM path — LLM calls not traced\nper agent in openai_events)",
                      ha="center", va="center", transform=ax_agent.transAxes,
                      fontsize=10, color="#888", linespacing=1.6)

    ax_agent.set_title("Chart 2a — Avg LLM Latency per Sub-Agent\n(agentic sessions only)",
                       fontweight="bold", pad=6, fontsize=10)
    ax_agent.yaxis.grid(True, alpha=0.4, zorder=0)
    ax_agent.set_axisbelow(True)
    ax_agent.spines["top"].set_visible(False)
    ax_agent.spines["right"].set_visible(False)

    # ── 2b: agents-per-turn complexity ────────────────────────────────────
    # Shows how much work each turn required — a direct pain-point signal.
    turn_nums   = [t.get("turn", i + 1) for i, t in enumerate(turns)]
    agent_counts = [len(t.get("agents_fired") or []) for t in turns]
    tool_counts  = [len(t.get("tool_sequence") or []) for t in turns]

    x = range(len(turns))
    bars_a = ax_turn.bar(x, agent_counts, color="#5B9BD5", label="Agents fired",
                         zorder=2, edgecolor="white", linewidth=0.5)
    if any(tc > 0 for tc in tool_counts):
        bars_t = ax_turn.bar(x, tool_counts, color=_COMP_COLOR,
                             label="Tools called", zorder=3,
                             edgecolor="white", linewidth=0.5)

    for xi, ac, tc in zip(x, agent_counts, tool_counts):
        top = max(ac, tc)
        ax_turn.text(xi, top + 0.1, str(ac), ha="center", va="bottom",
                     fontsize=9, color="#333", fontweight="bold")

    ax_turn.set_xticks(list(x))
    ax_turn.set_xticklabels([f"T{n}" for n in turn_nums], fontsize=10)
    ax_turn.set_xlabel("Turn", fontsize=10)
    ax_turn.set_ylabel("Count", fontsize=10)
    ax_turn.set_ylim(0, max(agent_counts + [1]) + 2)
    ax_turn.set_title("Chart 2b — Sub-Agents / Tools per Turn  (turn complexity)\n"
                      "taller bar = more agents invoked = heavier processing",
                      fontweight="bold", pad=6, fontsize=10)
    ax_turn.legend(fontsize=9, framealpha=0.8)
    ax_turn.yaxis.grid(True, alpha=0.4, zorder=0)
    ax_turn.set_axisbelow(True)
    ax_turn.spines["top"].set_visible(False)
    ax_turn.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3 — Intent × Tool-call timeline
# ═══════════════════════════════════════════════════════════════════════════

def _chart_intent_tools(ax: plt.Axes, turns: list[dict]) -> None:
    n = len(turns)
    if n == 0:
        ax.text(0.5, 0.5, "No turns", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")
    ax.set_title("Conversation Flow — Intent & Tool Calls per Turn",
                 fontweight="bold", pad=10)

    col_intent     = 0.0
    col_msg        = 0.18
    col_tools      = 0.52
    intent_w       = 0.16
    row_h          = 0.9   # height of each row in data-coords

    # Column headers
    header_y = n - 0.1
    for xpos, label in [(col_intent + intent_w / 2, "Intent"),
                        (col_msg + 0.15,            "User said"),
                        (col_tools + 0.22,          "Tools / Agents")]:
        ax.text(xpos, header_y, label, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#444")

    ax.axhline(n - 0.25, color="#ccc", linewidth=0.8, xmin=0, xmax=1)

    for i, turn in enumerate(turns):
        # rows drawn bottom-to-top: turn 1 at top → index n-1 at bottom
        row_y   = (n - 1 - i)
        box_bot = row_y - row_h / 2
        box_top = row_y + row_h / 2

        intent = (turn.get("intent") or "unrecognised").lower()
        color  = INTENT_COLORS.get(intent, "#BDC3C7")

        # ── intent badge ──
        fancy = mpatches.FancyBboxPatch(
            (col_intent, box_bot), intent_w, row_h,
            boxstyle="round,pad=0.01", linewidth=0.5,
            edgecolor="white", facecolor=color, zorder=2
        )
        ax.add_patch(fancy)
        ax.text(col_intent + intent_w / 2, row_y,
                f"T{turn.get('turn', i+1)}\n{intent.replace('_', ' ')}",
                ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", zorder=3)

        # ── user message (truncated) ──
        msg = (turn.get("user_message") or "")[:55]
        if len(turn.get("user_message") or "") > 55:
            msg += "…"
        ax.text(col_msg, row_y, msg,
                ha="left", va="center", fontsize=7.5, color="#333",
                wrap=False, zorder=2)

        # ── tools / agents ──
        tools = turn.get("tool_sequence") or []
        agents = turn.get("agents_fired") or []
        items = tools if tools else agents

        chip_x = col_tools
        chip_w = 0.085
        chip_h = 0.55
        gap    = 0.005
        for item in items:
            short = item.replace("_", "\n") if "_" in item else item
            c = TOOL_COLORS.get(item, _DEFAULT_TOOL_COLOR)
            chip = mpatches.FancyBboxPatch(
                (chip_x, row_y - chip_h / 2), chip_w, chip_h,
                boxstyle="round,pad=0.005", linewidth=0.4,
                edgecolor="white", facecolor=c, zorder=2
            )
            ax.add_patch(chip)
            ax.text(chip_x + chip_w / 2, row_y, short,
                    ha="center", va="center", fontsize=5.5, color="white",
                    fontweight="bold", zorder=3, linespacing=1.1)
            chip_x += chip_w + gap
            if chip_x > 0.97:
                break  # overflow guard

        # thin separator
        ax.axhline(box_bot, color="#eee", linewidth=0.5, xmin=0, xmax=1, zorder=1)

    # ── legend for intent colours ──
    legend_patches = [
        mpatches.Patch(facecolor=c, label=intent.replace("_", " ").title())
        for intent, c in INTENT_COLORS.items()
        if any((t.get("intent") or "").lower() == intent for t in turns)
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=7, ncol=3, framealpha=0.85,
                  title="Intent", title_fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def render(session_id: str, show: bool = False) -> Path:
    conv   = load_conversation(session_id)
    events = load_openai_events(session_id)
    turns  = conv.get("turns", [])

    token_data    = aggregate_tokens(events)
    latency_data  = aggregate_latency(events)
    orchestration = (turns[0].get("orchestration_mode", "fsm") if turns else "fsm")

    # ── figure layout ──────────────────────────────────────────────────────
    fig_h = max(10, 3 + len(turns) * 0.7)
    fig = plt.figure(figsize=(18, fig_h), facecolor="#FAFAFA")
    gs  = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1, 1, max(1, len(turns) * 0.28)],
        hspace=0.55, wspace=0.35,
        top=0.93, bottom=0.04, left=0.05, right=0.97,
    )

    ax_tokens = fig.add_subplot(gs[0, 0])
    ax_agent_lat = fig.add_subplot(gs[0, 1])
    ax_turn_lat  = fig.add_subplot(gs[1, :])
    ax_flow      = fig.add_subplot(gs[2, :])

    mode_label = "agentic" if orchestration == "agentic" else "fsm"
    fig.suptitle(
        f"Session analytics — {session_id}  [{mode_label}]",
        fontsize=14, fontweight="bold", y=0.97, color="#222"
    )

    _chart_agent_frequency(ax_tokens, turns, token_data)
    _chart_latency(ax_agent_lat, ax_turn_lat, latency_data, turns, token_data)
    _chart_intent_tools(ax_flow, turns)

    out_path = _LOGS / f"analytics_{session_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")

    if show:
        matplotlib.use("MacOSX")
        plt.switch_backend("MacOSX")
        plt.show()

    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-session trace analyser")
    parser.add_argument("--session", default=None,
                        help="session_id to analyse (default: latest log)")
    parser.add_argument("--show",    action="store_true",
                        help="open the chart in a window after saving")
    args = parser.parse_args()

    session_id = args.session or _latest_session_id()
    print(f"Analysing session: {session_id}")
    render(session_id, show=args.show)


if __name__ == "__main__":
    main()
