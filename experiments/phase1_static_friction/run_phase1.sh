#!/usr/bin/env bash
# Phase 1 - static heterogeneous terrain for lander touchdown.  Steps (run in this order; "all" runs every step):
#   soil       generate the soil maps on the lander's DTM (if missing) and the per-face friction overlay for /World/Terrain/Near2m
#   gate       PhysX gate test: per-face friction zones and friction combine modes (Isaac Sim)
#   touchdown  [speeds...]  300 Near2m sites x {uniform, heterogeneous} x speeds (default 1 2 3 m/s)  (Isaac Sim, ~15 min per case)
#   annotate   soil state under every footpad (lunar_regolith contact-query)
#   plot       comparison tables and figures
#   all        soil gate touchdown annotate plot
# Paths come from common/study_paths.py (.env.local / submodules).  Results: results/phase1/ (git-ignored); figures: docs/images/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(cd "$HERE/../.." && pwd)"
eval "$(python3 "$STUDY/common/study_paths.py")"
LOGS="$RESULTS_ROOT/phase1/logs"
mkdir -p "$LOGS"

step_soil() {
  echo "== soil: $REGOLITH_RUN_DIRECTORY"
  cd "$REGOLITH_REPOSITORY"
  if [ ! -f "$REGOLITH_RUN_DIRECTORY/run_metadata.json" ]; then
    [ -f data/nac_dem/NAC_DTM_APOLLO17_2M.TIF ] || "$REGOLITH_PYTHON" -m lunar_regolith fetch-dem --site apollo17_2m_mosaic
    "$REGOLITH_PYTHON" -m lunar_regolith generate --config "$REGOLITH_CONFIG" --run-name "$REGOLITH_RUN_NAME" --no-visualize
  fi
  "$REGOLITH_PYTHON" -m lunar_regolith contact-overlay --config "$REGOLITH_CONFIG" --run-dir "$REGOLITH_RUN_DIRECTORY" \
      --usd-scene "$LANDER_SCENE" --mesh-path /World/Terrain/Near2m --out "$RESULTS_ROOT/phase1/soil_contact" 2>&1 | grep -v "^Warning"
  cd "$STUDY"
}

step_gate() {
  echo "== gate (log $LOGS/friction_gate.log)"
  ISAAC_LOG="$LOGS/friction_gate.log" "$LANDER_ISAAC_RUNNER" "$HERE/verify_friction_zones_in_physx.py"
  grep -E "^Slab|^slab|honoured" "$LOGS/friction_gate.log"
}

step_touchdown() {
  local speeds=("$@")
  [ ${#speeds[@]} -eq 0 ] && speeds=(1 2 3)
  for speed in "${speeds[@]}"; do
    for mode in uniform heterogeneous; do
      local log="$LOGS/touchdown_${mode}_v${speed}.log"
      echo "== touchdown ${mode} ${speed} m/s (log $log)"
      ISAAC_LOG="$log" "$LANDER_ISAAC_RUNNER" "$HERE/heterogeneous_touchdown_test.py" --friction-mode "$mode" --touchdown-speed "$speed" --batch 25
      grep -E "^batch |^saved" "$log" | tail -2
    done
  done
}

step_annotate() { echo "== annotate"; "$REGOLITH_PYTHON" "$HERE/annotate_footpad_soil.py"; }
step_plot() { echo "== plot"; "$REGOLITH_PYTHON" "$HERE/plot_heterogeneous_touchdown.py"; }

case "${1:-}" in
  soil) step_soil ;;
  gate) step_gate ;;
  touchdown) shift; step_touchdown "$@" ;;
  annotate) step_annotate ;;
  plot) step_plot ;;
  all) step_soil; step_gate; step_touchdown; step_annotate; step_plot ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
