#!/usr/bin/env bash
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/run_helix_local_model.sh" kimi_k2_thinking
