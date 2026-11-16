#!/usr/bin/env python3
"""Render-ready appearance scene for one deformable-soil touchdown: field-driven regolith + the footpads' imprints
(plain Python with lunar_regolith + usd-core; Phase 2 visualisation).

    experiments/phase2_deformable/run_phase2.sh appearance-build        (or: <regolith python> build_touchdown_appearance.py --result ...)

Input: a soil_touchdown.py result recorded with --record-poses. For every footpad the imprint is a circle of the pad
radius at the pad's final position; its depth is the pad's final sinkage below the undeformed surface (pad in the
dent), or its permanent plastic sinkage if the pad has lifted - both from the pressure-sinkage model of the run. The wall
slopes at the local surface friction angle of the SYNTHETIC soil fields (lunar_regolith appearance assumption).

The render surface is the lander scene's Near2m tier itself (the surface the pads' sinkage was measured against),
refined exactly near the lander; Near2m is NOT modified - the render script hides it and shows these meshes instead.
Writes <out>/scene.usd (+ textures, materials), imprints.json and touchdown_appearance.json (pose, provenance).
All soil values are SYNTHETIC; the appearance mapping (lunar_regolith config/appearance.yaml) is an assumption.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

from lunar_regolith.appearance.fields import load_surface_fields  # noqa: E402
from lunar_regolith.appearance.geometry import GridSurface  # noqa: E402
from lunar_regolith.appearance.pipeline import SceneSpec, build_scene  # noqa: E402
from lunar_regolith.appearance.settings import load_appearance_config  # noqa: E402
from lunar_regolith.config import load_config  # noqa: E402
from lunar_regolith.contact.deformation import DeformationField, SurfaceImprint  # noqa: E402
from lunar_regolith.contact.soil_contact import SoilContactModel  # noqa: E402

PAD_RADIUS_M = 0.455


def imprint_depth(record, unloading_factor):
    """(visible dent depth, permanent depth) of one pad. ``deepest_plastic_sinkage_m`` is the deepest sinkage the pad
    reached (the start of its unloading line), not the permanent dent. A pad that touched the soil during the last second
    sits in its dent: depth = its final sinkage (for a lightly loaded pad that is where the unloading line reaches zero
    pressure, i.e. the dent floor). A lifted pad leaves the permanent dent, ~ deepest x (1 - 1 / unloading_factor)
    (exact for a linear loading curve; the lunar curves here have n = 1)."""
    deepest = record["deepest_plastic_sinkage_m"] or 0.0
    permanent = deepest * (1.0 - 1.0 / unloading_factor)
    final = record["final_sinkage_m"] or 0.0
    loaded = (record.get("loaded_fraction_of_steps_last_second") or 0.0) > 0.0 and final > 0
    return (final if loaded else permanent), permanent


def main():
    """Build the scene for every landing in the result file (normally one)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--result", type=Path, required=True, help="soil_touchdown.py JSON recorded with --record-poses")
    parser.add_argument("--preset", default="offline")
    parser.add_argument("--half-size", type=float, default=15.0, help="refined render patch half width around the lander (m)")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    from pxr import Usd

    result = json.loads(arguments.result.read_text())
    soil_mode = result["args"]["soil_mode"]
    configuration = load_config(study_paths.REGOLITH_REPOSITORY / study_paths.REGOLITH_CONFIG)
    appearance = load_appearance_config(study_paths.REGOLITH_REPOSITORY / "config" / "appearance.yaml")
    lon, lat = configuration.contact.site_lon_deg, configuration.contact.site_lat_deg
    stage = Usd.Stage.Open(str(study_paths.LANDER_SCENE))
    stage.GetPrimAtPath("/World").GetVariantSets().GetVariantSet("terrain").SetVariantSelection("nac_2m")
    near = stage.GetPrimAtPath("/World/Terrain/Near2m")
    grid = near.GetCustomDataByKey("grid")
    surface = GridSurface.from_usd_grid_mesh(near, grid["x0"], grid["y0"], grid["dx"], grid["dy"], grid["nx"], grid["ny"])
    half_extent = (grid["nx"] - 1) * grid["dx"] / 2
    fields = load_surface_fields(study_paths.REGOLITH_RUN_DIRECTORY, configuration, lon, lat, (0.0, 0.0), half_extent + 4.0)
    fields = fields.crop((-half_extent, half_extent), (-half_extent, half_extent))
    soil = SoilContactModel.from_run_dir(study_paths.REGOLITH_RUN_DIRECTORY, configuration, lon, lat,
                                         configuration.contact.interface_friction)
    unloading_factor = float(result["args"]["sinkage_parameters"]["unloading_factor"])
    for landing in result["results"]:
        if "pose_series" not in landing:
            raise SystemExit("the result has no pose_series - run soil_touchdown.py with --record-poses")
        imprints = []
        for pad, record in landing["pads"].items():
            east, north = record["final_point"][:2]
            depth, permanent = imprint_depth(record, unloading_factor)
            phi = float(np.asarray(soil.query_local(np.array([east]), np.array([north]), 0.0)["friction_angle_deg"])[0])
            imprints.append(SurfaceImprint(east_m=east, north_m=north, radius_m=PAD_RADIUS_M, sinkage_m=depth, plastic_sinkage_m=permanent,
                                           wall_angle_deg=phi if np.isfinite(phi) else None,
                                           source=f"pad {pad}: final sinkage {record['final_sinkage_m']:.4f} m, deepest "
                                                  f"{record['deepest_plastic_sinkage_m']:.4f} m (soil_touchdown.py, {soil_mode} soil)"))
        d = appearance.deformation
        deformation = DeformationField(imprints, rim_volume_fraction=d.rim_volume_fraction, rim_width_factor=d.rim_width_factor,
                                       compaction_depth_factor=d.compaction_depth_factor,
                                       default_wall_angle_deg=d.wall_angle_deg or 40.0,
                                       metadata={"result": str(arguments.result), "site": landing["idx"]})
        out = arguments.out or study_paths.phase_results_directory(
            "phase2", "appearance", f"site{landing['idx']}_{soil_mode}_v{landing['touchdown_speed_target']:g}")
        out.mkdir(parents=True, exist_ok=True)
        pose = landing["pose_series"][-1]
        spec = SceneSpec(f"touchdown_site{landing['idx']}", fields, (pose[1], pose[2]), arguments.half_size,
                         f"Apollo LM at rest after a {landing['touchdown_speed_target']:g} m/s touchdown on {soil_mode} soil (Phase 2)",
                         deformation=deformation, include_collider=False, surface=surface)
        meta = build_scene(spec, configuration, appearance, arguments.preset, out)
        deformation.save(out / "imprints.json")
        info = dict(scene=str(out / "scene.usd"), site_index=landing["idx"], soil_mode=soil_mode,
                    touchdown_speed=landing["touchdown_speed_target"], final_pose=pose, tilt_deg=landing["tilt_deg"],
                    pads={k: dict(final_point=p["final_point"], final_sinkage_m=p["final_sinkage_m"],
                                  plastic_sinkage_m=p["deepest_plastic_sinkage_m"], final_load_share=p["final_load_share"])
                          for k, p in landing["pads"].items()},
                    imprint_render_check=meta["imprint_render_check"], preset=arguments.preset,
                    surface="lander scene Near2m (unchanged; render meshes refine it exactly)",
                    note="SYNTHETIC soil fields; appearance mapping = assumption; imprint depths from the Phase 2 pressure-sinkage run",
                    provenance=study_paths.git_provenance())
        (out / "touchdown_appearance.json").write_text(json.dumps(info, indent=1))
        print(f"site {landing['idx']} {soil_mode}: {out / 'scene.usd'}  imprints " + ", ".join(
            f"{k} {100 * i.sinkage_m:.1f} cm" for k, i in zip(landing["pads"], imprints))
              + f"  (rendered floor = physical depth: max error "
                f"{max(c['max_abs_render_minus_physical_m'] for c in meta['imprint_render_check']):.1e} m)")


if __name__ == "__main__":
    main()
