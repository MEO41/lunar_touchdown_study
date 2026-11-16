#!/usr/bin/env bash
# Phase 2 - deformable regolith.
#   select          sites where exactly one pad stands over a shallow soft pocket (from the Phase 1 soil annotation)
#   columns         soil columns around each of those sites: heterogeneous (as generated) and a uniform control
#   soil-touchdown  [speeds...]  Isaac Sim: per-footpad pressure-sinkage + shear (lunar_regolith.contact.sinkage), both soil modes (default 1 2 3)
#   soil-plot       comparison table and figure
#   appearance      [site] [speed] [mode]  re-run one soil touchdown with poses, build its lunar_regolith appearance scene
#                   (field-driven regolith, footpad imprints) and render it in Isaac Sim (default: site 20, 3 m/s, heterogeneous)
#   preview steps (compliant-contact illustration, ILLUSTRATIVE stiffness):  simulate [speed] | render | plot
#   all             select columns soil-touchdown soil-plot
# Requires Phase 1 results (run_phase1.sh soil touchdown annotate).  Results: results/phase2/ (git-ignored).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(cd "$HERE/../.." && pwd)"
eval "$(python3 "$STUDY/common/study_paths.py")"
LOGS="$RESULTS_ROOT/phase2/logs"
DEMO="$RESULTS_ROOT/phase2/soft_pocket_demo"
TOUCHDOWN="$STUDY/experiments/phase1_static_friction/heterogeneous_touchdown_test.py"
mkdir -p "$LOGS" "$DEMO"

step_select() { echo "== select"; "$REGOLITH_PYTHON" "$HERE/select_soft_pocket_site.py" 2>&1 | grep -v "^Warning"; }

step_simulate() {
  local speed="${1:-1}"
  local site
  site="$(python3 -c "import json; print(json.load(open('$DEMO/soft_pocket_site.json'))['site']['site_index'])")"
  echo "== simulate site $site at $speed m/s: soft pocket, then rigid"
  ISAAC_LOG="$LOGS/soft_pocket_v${speed}.log" "$LANDER_ISAAC_RUNNER" "$TOUCHDOWN" --friction-mode uniform --touchdown-speed "$speed" --sites "$site" \
      --soft-pocket "$DEMO/soft_pocket_site.json" --record-poses --max-seconds 20 --out "$DEMO/touchdown_soft_pocket_v${speed}.json"
  ISAAC_LOG="$LOGS/rigid_v${speed}.log" "$LANDER_ISAAC_RUNNER" "$TOUCHDOWN" --friction-mode uniform --touchdown-speed "$speed" --sites "$site" \
      --record-poses --max-seconds 20 --out "$DEMO/touchdown_rigid_v${speed}.json"
  grep -hE "^saved" "$LOGS/soft_pocket_v${speed}.log" "$LOGS/rigid_v${speed}.log"
}

step_render() {
  echo "== render (log $LOGS/render_soft_pocket.log)"
  ISAAC_LOG="$LOGS/render_soft_pocket.log" "$LANDER_ISAAC_RUNNER" "$HERE/render_soft_pocket_touchdown.py"
  grep -E "^rendered|^saved" "$LOGS/render_soft_pocket.log"
}

step_plot() { echo "== plot"; "$REGOLITH_PYTHON" "$HERE/plot_soft_pocket_touchdown.py"; }

step_columns() { echo "== columns"; "$REGOLITH_PYTHON" "$HERE/prepare_soil_columns.py" 2>&1 | grep -v "^Warning"; }

step_soil_touchdown() {
  local speeds=("$@")
  [ ${#speeds[@]} -eq 0 ] && speeds=(1 2 3)
  for speed in "${speeds[@]}"; do
    for mode in heterogeneous uniform; do
      local log="$LOGS/soil_touchdown_${mode}_v${speed}.log"
      echo "== soil touchdown ${mode} ${speed} m/s (log $log)"
      ISAAC_LOG="$log" "$LANDER_ISAAC_RUNNER" "$HERE/soil_touchdown.py" --soil-mode "$mode" --touchdown-speed "$speed" --max-seconds 40
      grep -E "^site|^saved" "$log"
    done
  done
}

step_soil_plot() { echo "== soil plot"; "$REGOLITH_PYTHON" "$HERE/plot_soil_touchdown.py"; }

step_appearance() {
  local site="${1:-20}" speed="${2:-3}" mode="${3:-heterogeneous}"
  local dir="$RESULTS_ROOT/phase2/appearance"
  local poses="$dir/soil_touchdown_${mode}_v${speed}_site${site}_poses.json"
  mkdir -p "$dir"
  echo "== appearance: site $site, $speed m/s, $mode soil - touchdown with poses (log $LOGS/appearance_touchdown.log)"
  ISAAC_LOG="$LOGS/appearance_touchdown.log" "$LANDER_ISAAC_RUNNER" "$HERE/soil_touchdown.py" --soil-mode "$mode" \
      --touchdown-speed "$speed" --sites "$site" --max-seconds 40 --record-poses --out "$poses"
  echo "== appearance: build scene"
  "$REGOLITH_PYTHON" "$HERE/build_touchdown_appearance.py" --result "$poses" 2>&1 | grep -v "^INFO"
  echo "== appearance: render (log $LOGS/appearance_render.log)"
  ISAAC_LOG="$LOGS/appearance_render.log" "$LANDER_ISAAC_RUNNER" "$HERE/render_touchdown_appearance.py" \
      --case "$dir/site${site}_${mode}_v${speed}"
  grep -E "^rendered|^contact sheet" "$LOGS/appearance_render.log"
}

case "${1:-}" in
  select) step_select ;;
  simulate) shift; step_simulate "$@" ;;
  render) step_render ;;
  plot) step_plot ;;
  columns) step_columns ;;
  soil-touchdown) shift; step_soil_touchdown "$@" ;;
  soil-plot) step_soil_plot ;;
  appearance) shift; step_appearance "$@" ;;
  all) step_select; step_columns; step_soil_touchdown; step_soil_plot ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
