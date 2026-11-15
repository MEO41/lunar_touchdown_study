#!/usr/bin/env python3
"""Add the soil state under every footpad to heterogeneous-touchdown results (plain Python; runs lunar_regolith in its own environment).

    experiments/phase1_static_friction/run_phase1.sh annotate          (all touchdown results)
    <regolith python> annotate_footpad_soil.py results/phase1/touchdown/heterogeneous_touchdown_heterogeneous_v2.json [...]

For each landing and footpad, the pad's final contact point (site frame: x east, y north, origin = touchdown point) is sent to
`lunar_regolith contact-query`, which averages the SYNTHETIC soil fields over the footpad disk (0.91 m) and returns: dominant facies, friction
angle, bulk density, cohesion, compaction, interface friction, the share of the pad over a soft subsurface pocket and that pocket's depth window.
Pockets do not act in Phase 1 (rigid terrain); they mark which pads would sink first in Phase 2.  Writes <input>_soil.json next to each input.

Repository, Python and soil configuration come from common/study_paths.py (REGOLITH_* settings).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

LUNAR_REGOLITH_DIRECTORY = study_paths.REGOLITH_REPOSITORY
LUNAR_REGOLITH_PYTHON = study_paths.REGOLITH_PYTHON
LUNAR_REGOLITH_CONFIG = study_paths.REGOLITH_CONFIG
LUNAR_REGOLITH_RUN_DIRECTORY = str(study_paths.REGOLITH_RUN_DIRECTORY)
KEPT_SOIL_FIELDS = ("dominant_facies", "facies_fractions", "friction_angle_deg", "bulk_density_kg_m3", "cohesion_pa", "compaction_index",
                    "interface_mu_static", "interface_mu_dynamic", "soft_lens_area_fraction", "shallowest_soft_lens_top_m",
                    "deepest_soft_lens_bottom_m", "regolith_thickness_min_m", "valid_fraction", "synthetic", "calibration_status")


def footpad_points(results):
    """One query point per landing and footpad: its final contact position in the site frame."""
    points = []
    for landing_number, landing in enumerate(results):
        for pad_name, pad in landing["pads"].items():
            east, north = pad["final_contact_east_north"]
            points.append({"landing": landing_number, "pad": pad_name, "east": east, "north": north})
    return points


def query_soil(points):
    """Run `lunar_regolith contact-query` on the points and return its per-point results (same order)."""
    with tempfile.TemporaryDirectory() as directory:
        points_path, output_path = Path(directory) / "points.json", Path(directory) / "soil.json"
        points_path.write_text(json.dumps(points))
        command = [str(LUNAR_REGOLITH_PYTHON), "-m", "lunar_regolith", "contact-query", "--config", LUNAR_REGOLITH_CONFIG,
                   "--run-dir", LUNAR_REGOLITH_RUN_DIRECTORY, "--points", str(points_path), "--out", str(output_path)]
        completed = subprocess.run(command, cwd=LUNAR_REGOLITH_DIRECTORY, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(f"contact-query failed ({completed.returncode}):\n{completed.stderr[-2000:]}")
        return json.loads(output_path.read_text())


def annotate(result_path):
    """Annotate one results file and write <name>_soil.json; returns the output path."""
    data = json.loads(Path(result_path).read_text())
    points = footpad_points(data["results"])
    annotated = query_soil(points)
    for entry in annotated:
        soil = {k: entry["soil"][k] for k in KEPT_SOIL_FIELDS}
        data["results"][entry["landing"]]["pads"][entry["pad"]]["soil"] = soil
    data["args"]["soil_annotation"] = dict(lunar_regolith_config=LUNAR_REGOLITH_CONFIG, run_directory=LUNAR_REGOLITH_RUN_DIRECTORY,
                                           patch="footpad disk r = 0.455 m", note="SYNTHETIC / uncalibrated soil fields")
    output_path = Path(result_path).with_name(Path(result_path).stem + "_soil.json")
    output_path.write_text(json.dumps(data))
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        found = sorted(p for p in (study_paths.RESULTS_ROOT / "phase1" / "touchdown").glob("heterogeneous_touchdown_*_v*.json")
                       if not p.stem.endswith("_soil"))
        if not found:
            raise SystemExit(__doc__)
        sys.argv += [str(p) for p in found]
    for path in sys.argv[1:]:
        print("wrote", annotate(path))
