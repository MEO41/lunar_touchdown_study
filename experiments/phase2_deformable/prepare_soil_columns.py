#!/usr/bin/env python3
"""Soil columns around each Phase 2 site, for the deformable-soil touchdowns (plain Python with lunar_regolith installed).

    experiments/phase2_deformable/run_phase2.sh columns

For every candidate site in results/phase2/soft_pocket_demo/soft_pocket_site.json (one pad over a shallow soft pocket, three clear) this
writes two lunar_regolith SoilColumnGrids (local site frame x depth, see lunar_regolith.contact.sinkage):
  results/phase2/soil_columns/heterogeneous/site_<index>.npz   the SYNTHETIC soil fields as generated (pockets, layers, lateral variation)
  results/phase2/soil_columns/uniform/site_<index>.npz         control: every column replaced by the site's median POCKET-FREE column
                                                               (same depth profile everywhere, no pocket) - so the two soil modes differ
                                                               only in lateral heterogeneity, as in Phase 1's friction comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

from lunar_regolith.config import load_config  # noqa: E402
from lunar_regolith.contact.pipeline import soil_column_grid  # noqa: E402
from lunar_regolith.contact.sinkage import STATUS_INSIDE_REGOLITH, SoilColumnGrid  # noqa: E402

SITE_FILE = study_paths.RESULTS_ROOT / "phase2" / "soft_pocket_demo" / "soft_pocket_site.json"
OUTPUT = study_paths.phase_results_directory("phase2", "soil_columns")
HALF_SIZE_M, SPACING_M = 12.0, 0.5
POCKET_FREE_WEIGHT = 0.05            # a column is pocket-free when its soft weight stays below this at every depth


def uniform_control(grid):
    """Copy of `grid` whose columns all equal the median pocket-free column (property by property, depth by depth)."""
    pocket_free = np.nanmax(np.nan_to_num(grid.fields["soft_weight"]), axis=2) < POCKET_FREE_WEIGHT
    if pocket_free.sum() < 10:
        raise SystemExit("fewer than 10 pocket-free columns around the site - enlarge HALF_SIZE_M")
    fields = {name: np.broadcast_to(np.nanmedian(values[pocket_free], axis=0), values.shape).copy() for name, values in grid.fields.items()}
    status_profile = np.where((grid.status[pocket_free] == STATUS_INSIDE_REGOLITH).mean(axis=0) >= 0.5, STATUS_INSIDE_REGOLITH, 1)
    status = np.broadcast_to(status_profile.astype(np.int8), grid.status.shape).copy()
    metadata = dict(grid.metadata, control="uniform: median pocket-free column everywhere", pocket_free_columns=int(pocket_free.sum()))
    return SoilColumnGrid(grid.east, grid.north, grid.depth, fields, status, metadata)


def main():
    """Export heterogeneous and uniform column grids for every candidate site."""
    configuration = load_config(study_paths.REGOLITH_REPOSITORY / study_paths.REGOLITH_CONFIG)
    candidates = json.loads(SITE_FILE.read_text())["candidates"]
    for candidate in candidates:
        index = candidate["site_index"]
        east, north = candidate["site_east_north"]
        heterogeneous_path = OUTPUT / "heterogeneous" / f"site_{index}.npz"
        heterogeneous_path.parent.mkdir(parents=True, exist_ok=True)
        info = soil_column_grid(configuration, study_paths.REGOLITH_RUN_DIRECTORY, heterogeneous_path, east, north, HALF_SIZE_M, SPACING_M)
        uniform_path = OUTPUT / "uniform" / f"site_{index}.npz"
        uniform_path.parent.mkdir(parents=True, exist_ok=True)
        uniform_control(SoilColumnGrid.load(heterogeneous_path)).save(uniform_path)
        print(f"site {index}: {info['shape']} columns, {100 * info['soft_column_fraction']:.0f}% with a soft pocket -> {heterogeneous_path.name} "
              f"(+ uniform control)")


if __name__ == "__main__":
    main()
