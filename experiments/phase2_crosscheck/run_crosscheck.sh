#!/usr/bin/env bash
# Phase 2 cross-check with Project Chrono SCM (needs CHRONO_PYTHON in .env.local: PyChrono with the vehicle module, conda channel projectchrono).
#   export     terrain patch, SCM soil maps and lander description per Phase 2 site (lunar_regolith Python)
#   plate      plate-penetration test: SCM vs lunar_regolith's pressure-sinkage law, reference lunar soil
#   touchdown  the 30 Phase 2 touchdowns (5 sites x 1/2/3 m/s x heterogeneous/uniform) in SCM
#   compare    Chrono vs Isaac table and figure
#   all        export plate touchdown compare
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(cd "$HERE/../.." && pwd)"
eval "$(python3 "$STUDY/common/study_paths.py")"
[ -n "${CHRONO_PYTHON:-}" ] || { echo "set CHRONO_PYTHON in .env.local"; exit 2; }
LOGS="$RESULTS_ROOT/phase2/logs"; mkdir -p "$LOGS"

step_export() { echo "== export"; "$REGOLITH_PYTHON" "$HERE/export_chrono_inputs.py" 2>&1 | grep -v "^Warning"; }
step_plate() { echo "== plate"; "$CHRONO_PYTHON" "$HERE/chrono_plate_test.py"; }
step_touchdown() { echo "== touchdown (log $LOGS/chrono_touchdown.log)"; "$CHRONO_PYTHON" "$HERE/chrono_scm_touchdown.py" "$@" 2>&1 | tee "$LOGS/chrono_touchdown.log" | grep -E "site|saved"; }
step_compare() { echo "== compare"; "$REGOLITH_PYTHON" "$HERE/compare_chrono_isaac.py"; }

case "${1:-}" in
  export) step_export ;;
  plate) step_plate ;;
  touchdown) shift; step_touchdown "$@" ;;
  compare) step_compare ;;
  all) step_export; step_plate; step_touchdown; step_compare ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
