#!/usr/bin/env python3
"""Pick a landing site where exactly ONE footpad stands over a shallow soft pocket, and mark the terrain faces above that pocket
(plain Python with lunar_regolith installed; Phase 2 preview).

    experiments/phase2_deformable/run_phase2.sh select        (or: <regolith python> select_soft_pocket_site.py [--rank N])

Candidates come from the Phase 1 rigid, uniform-friction 1 m/s touchdowns with the soil annotation (annotate_footpad_soil.py): the
footpads' final positions there are where this lander actually stands on rigid ground.  A candidate has one pad with soft-pocket area
fraction >= 0.9 whose pocket starts at most --max-pocket-top-depth below the surface, the other three pads with no pocket at all, a
gentle site slope and an upright Phase 1 outcome (so any tilt in the soft-pocket run comes from the pocket, not from the slope).

For the chosen site, every Near2m terrain face within --radius of it whose centroid lies over a soft pocket (horizontal pocket weight
>= 0.5, the same definition as the soil annotation) is listed.  touchdown_with_soft_pocket.py makes exactly those faces compliant.

The pocket stiffness written here is ILLUSTRATIVE (chosen to make the pocket visibly give way, see POCKET_* below).  It is NOT derived from
the soil fields yet - deriving it from pressure-sinkage soil mechanics is the Phase 2 work in lunar_regolith.contact.  The pocket itself
comes from SYNTHETIC, uncalibrated soil fields.  Writes results/phase2/soft_pocket_demo/soft_pocket_site.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

from lunar_regolith.config import load_config  # noqa: E402
from lunar_regolith.contact.soil_contact import SoilContactModel  # noqa: E402

PHASE1_TOUCHDOWN = study_paths.RESULTS_ROOT / "phase1" / "touchdown" / "heterogeneous_touchdown_uniform_v1_soil.json"
FACES = study_paths.RESULTS_ROOT / "phase1" / "soil_contact" / "soil_friction_faces_World_Terrain_Near2m.npz"
OUTPUT_DIRECTORY = study_paths.phase_results_directory("phase2", "soft_pocket_demo")

# ILLUSTRATIVE compliant-contact parameters of the pocket faces (PhysX applies them per contact POINT; a footpad touches the mesh at several
# points).  The lander repository calibrated 5000 N/m / 1700 N s/m per point for the ordinary top layer (Apollo 11: 3.7 cm deep).  The pocket
# uses about an eighth of that stiffness, so a pad carrying a quarter of the lander's lunar weight (2.9 kN) settles a few decimetres deep, with
# a damping ratio near 1.  A rigid stop POCKET_STOP_DEPTH_M below the surface stands for the firm material under the pocket.
POCKET_STIFFNESS_PER_CONTACT = 600.0        # N/m
POCKET_DAMPING_PER_CONTACT = 700.0          # N s/m
POCKET_STOP_DEPTH_M = 0.5                   # m


def candidate_sites(results, max_pocket_top_depth, max_slope):
    """Landings with exactly one pad over a shallow pocket and three pads clear of any pocket, gentlest slope first."""
    candidates = []
    for landing in results:
        pads = landing["pads"]
        over = [k for k, p in pads.items() if p["soil"]["soft_lens_area_fraction"] >= 0.9
                and (p["soil"]["shallowest_soft_lens_top_m"] or 99) <= max_pocket_top_depth]
        clear = [k for k, p in pads.items() if p["soil"]["soft_lens_area_fraction"] == 0.0]
        upright = landing["tilt_deg"] <= landing["slope_deg"] + 5 and len(landing["touching"]) >= 3
        if len(over) == 1 and len(clear) == 3 and landing["slope_deg"] <= max_slope and upright:
            candidates.append(dict(site_index=landing["idx"], slope_deg=landing["slope_deg"], yaw_deg=landing["yaw_deg"],
                                   pocket_pad=over[0], pocket_pad_east_north=pads[over[0]]["final_contact_east_north"],
                                   pocket_top_m=pads[over[0]]["soil"]["shallowest_soft_lens_top_m"],
                                   pocket_bottom_m=pads[over[0]]["soil"]["deepest_soft_lens_bottom_m"],
                                   pocket_pad_soil=pads[over[0]]["soil"],
                                   other_pads_soil={k: pads[k]["soil"]["dominant_facies"] for k in clear},
                                   site_east_north=[landing["x"], landing["y"]]))
    return sorted(candidates, key=lambda c: (c["slope_deg"], c["pocket_top_m"]))


def pocket_faces(model, site_east_north, radius):
    """Indices (and pocket depth window) of the Near2m faces within `radius` of the site whose centroid lies over a soft pocket."""
    faces = np.load(FACES)
    east, north = faces["centroid_east"].astype(np.float64), faces["centroid_north"].astype(np.float64)
    near = np.flatnonzero(np.hypot(east - site_east_north[0], north - site_east_north[1]) <= radius)
    soil = model.query_local(east[near], north[near], 0.0)
    weight = np.nan_to_num(soil["soft_weight_xy"])
    over = weight >= 0.5
    return dict(face_indices=near[over].tolist(), pocket_weight=weight[over].round(3).tolist(),
                pocket_top_m=np.asarray(soil["soft_lens_top_m"])[over].round(3).tolist(),
                pocket_thickness_m=np.asarray(soil["soft_lens_thickness_m"])[over].round(3).tolist(),
                faces_examined=int(near.size))


def main():
    """Choose the site, list its pocket faces and write soft_pocket_site.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rank", type=int, default=0, help="which candidate (0 = gentlest slope)")
    parser.add_argument("--max-pocket-top-depth", type=float, default=0.7, help="pocket must start within this depth (m)")
    parser.add_argument("--max-slope", type=float, default=5.0, help="site slope limit (deg)")
    parser.add_argument("--radius", type=float, default=25.0, help="mark pocket faces within this distance of the site (m)")
    arguments = parser.parse_args()

    results = json.loads(PHASE1_TOUCHDOWN.read_text())["results"]
    candidates = candidate_sites(results, arguments.max_pocket_top_depth, arguments.max_slope)
    if not candidates:
        raise SystemExit("no candidate site - relax --max-pocket-top-depth / --max-slope")
    print(f"{len(candidates)} candidate sites (one pad over a pocket starting <= {arguments.max_pocket_top_depth} m, three clear, "
          f"slope <= {arguments.max_slope} deg):")
    for rank, c in enumerate(candidates[:8]):
        print(f"  rank {rank}: site {c['site_index']:3d}  slope {c['slope_deg']:4.1f} deg  pocket under {c['pocket_pad']}  "
              f"top {c['pocket_top_m']:.2f} m  bottom {c['pocket_bottom_m']:.2f} m")
    chosen = candidates[arguments.rank]

    configuration = load_config(study_paths.REGOLITH_REPOSITORY / study_paths.REGOLITH_CONFIG)
    model = SoilContactModel.from_run_dir(study_paths.REGOLITH_RUN_DIRECTORY, configuration, configuration.contact.site_lon_deg,
                                          configuration.contact.site_lat_deg, configuration.contact.interface_friction)
    faces = pocket_faces(model, chosen["site_east_north"], arguments.radius)
    output = dict(
        site=chosen, pocket=faces,
        compliance=dict(stiffness_per_contact_N_per_m=POCKET_STIFFNESS_PER_CONTACT, damping_per_contact_N_s_per_m=POCKET_DAMPING_PER_CONTACT,
                        stop_depth_m=POCKET_STOP_DEPTH_M, status="ILLUSTRATIVE - chosen to make the pocket visibly give way; not derived "
                        "from soil mechanics yet (Phase 2 work)"),
        selection=vars(arguments), source=str(PHASE1_TOUCHDOWN),
        note="Soft pocket from SYNTHETIC, uncalibrated soil fields (lunar_regolith); terrain surface = LROC NAC DTM (unchanged).",
        provenance=study_paths.git_provenance())
    path = OUTPUT_DIRECTORY / "soft_pocket_site.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"chosen site {chosen['site_index']} (pocket under pad {chosen['pocket_pad']}): {len(faces['face_indices'])} pocket faces of "
          f"{faces['faces_examined']} within {arguments.radius:g} m -> {path}")


if __name__ == "__main__":
    main()
