#!/usr/bin/env bash
# run_demo.sh — one-command end-to-end voice demo.
# Boots the Pipecat WebRTC voice bot. Open the printed URL, click Connect, allow
# the mic, and talk to the agent (English / Hindi / Hinglish).
#
#   ./run_demo.sh
#
set -euo pipefail

cd "$(dirname "$0")"

PY="../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

echo "🚀 Starting Insurance Sales Agent voice demo…"
echo "   When it says 'Bot ready', open:  http://localhost:7860"
echo "   (click Connect, allow microphone, then speak)"
echo

exec "$PY" -m voice.run_voice
