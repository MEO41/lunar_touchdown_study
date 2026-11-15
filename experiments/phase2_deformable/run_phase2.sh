#!/usr/bin/env bash
# Phase 2 - deformable regolith.  Steps so far (preview: one footpad sinking into a soft pocket):
#   select    pick a site where exactly one pad stands over a shallow soft pocket; list the terrain faces above the pocket
#   simulate  [speed]  that site twice in Isaac Sim: pocket faces compliant (ILLUSTRATIVE stiffness) and all-rigid (default 1 m/s)
#   render    path-traced sequence of the sinking pad and the rigid comparison (docs/images/soft_pocket_touchdown_sequence.jpg)
#   plot      pad depth and lander tilt over time (docs/images/soft_pocket_touchdown_timeseries.png)
#   all       select simulate render plot
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

case "${1:-}" in
  select) step_select ;;
  simulate) shift; step_simulate "$@" ;;
  render) step_render ;;
  plot) step_plot ;;
  all) step_select; step_simulate; step_render; step_plot ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
