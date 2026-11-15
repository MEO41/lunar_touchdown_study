#!/usr/bin/env python3
"""Inputs for the Project Chrono SCM cross-check of the Phase 2 deformable-soil touchdowns (plain Python with lunar_regolith + usd-core).

    experiments/phase2_crosscheck/run_crosscheck.sh export

Why: Phase 2 couples lunar_regolith's pressure-sinkage model to the lander in Isaac Sim through our own per-footpad forces.  Chrono's
SCM (Soil Contact Model) implements the same soil laws (Bekker-Wong pressure-sinkage, Janosi shear) in an independent engine with a
deforming terrain grid.  Running the same landings there checks our coupling.

Per site (default: the Phase 2 candidate sites), written to results/phase2/chrono_inputs/site_<index>/:
  terrain.obj          Near2m terrain patch (2 m grid, same triangulation as the lander scene), scene coordinates (x east, y north, z up).
                       Chrono deforms ITS OWN copy; the source DTM is never modified.
  soil_<mode>.npz      SCM soil maps on the soil-column grid (east, north) for mode heterogeneous / uniform:
                         bekker_kphi_pa_per_m  first-loading stiffness of OUR layered model for a footpad-size plate at REFERENCE_SINKAGE_M
                                               (SCM uses p = (Kc/b + Kphi) z^n; we set Kc = 0, n = 1, so p = Kphi z: a linearisation of
                                               our curve - exact for homogeneous soil, approximate where the layers change with depth)
                         cohesion_pa, friction_angle_deg   shallow soil strength (Janosi / Mohr-Coulomb), as in our model
                         elastic_k_pa_per_m    unloading stiffness = unloading_factor x Kphi (our model)
                         damping_pa_s_per_m    damping_time_constant x elastic_k (our model on the unloading line)
                       janosi_k_m and the plate radius are stored as scalars.
  lander.json          mass, centre of mass, inertia, footpad bottom centres (lander frame), pad radius, site pose and the release
                       state for each touchdown speed (identical to soil_touchdown.py).
All soil values derive from SYNTHETIC soil fields (lunar_regolith) and assumed model constants; label results accordingly.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402
from landing_geometry import TerrainGridSurface, footpad_geometry  # noqa: E402

from lunar_regolith.contact.sinkage import PressureSinkageField, SinkageParameters, SoilColumnGrid  # noqa: E402

COLUMNS = study_paths.RESULTS_ROOT / "phase2" / "soil_columns"
OUTPUT = study_paths.phase_results_directory("phase2", "chrono_inputs")
REFERENCE_SINKAGE_M = 0.05
PAD_RADIUS_M = 0.455
GRAVITY = 1.62
CLEARANCE_M = 0.25
DAMPING_TIME_CONSTANT_S = 0.003           # as in soil_touchdown.py


def open_scene():
    """The lander scene with the nac_2m terrain variant selected (usd-core)."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(study_paths.LANDER_SCENE))
    stage.GetPrimAtPath("/World").GetVariantSets().GetVariantSet("terrain").SetVariantSelection("nac_2m")
    return stage


def write_terrain_patch(surface, center, half_size, path):
    """Near2m grid triangles within the square around `center` as a Wavefront OBJ (builder's triangulation)."""
    i0 = int((center[0] - half_size - surface.x0) // surface.dx)
    i1 = int((center[0] + half_size - surface.x0) // surface.dx) + 1
    j0 = int((center[1] - half_size - surface.y0) // surface.dy)
    j1 = int((center[1] + half_size - surface.y0) // surface.dy) + 1
    lines, index = [], {}
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            index[(i, j)] = len(index) + 1
            lines.append(f"v {surface.x0 + i * surface.dx:.4f} {surface.y0 + j * surface.dy:.4f} {surface.heights[j, i]:.4f}")
    for j in range(j0, j1):
        for i in range(i0, i1):
            a, b, c, d = index[(i, j)], index[(i + 1, j)], index[(i, j + 1)], index[(i + 1, j + 1)]
            lines += [f"f {a} {b} {d}", f"f {a} {d} {c}"]          # same split as TerrainGridSurface.height (u >= v: a-b-d)
    path.write_text("# Near2m terrain patch (LROC NAC DTM via apollo_lander_modeling), scene coordinates, metres\n" + "\n".join(lines) + "\n")


def soil_maps(grid_path, parameters):
    """SCM parameter maps from one soil-column grid (see module docstring)."""
    grid = SoilColumnGrid.load(grid_path)
    field = PressureSinkageField(grid, PAD_RADIUS_M, parameters)
    shape = (grid.north.size, grid.east.size)
    kphi, cohesion, friction = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            sinkage, pressure, tangent = field.column_curve(i, j)
            kphi[i, j] = np.interp(REFERENCE_SINKAGE_M, sinkage, pressure) / REFERENCE_SINKAGE_M      # secant to the reference sinkage
            cohesion[i, j], friction[i, j] = field.strength(grid.east[j], grid.north[i], 0.0)
    elastic = parameters.unloading_factor * kphi
    return dict(east=grid.east, north=grid.north, bekker_kphi_pa_per_m=kphi, bekker_kc=np.zeros(shape), bekker_n=1.0,
                cohesion_pa=cohesion, friction_angle_deg=friction, janosi_k_m=parameters.shear_deformation_modulus_m,
                elastic_k_pa_per_m=elastic, damping_pa_s_per_m=DAMPING_TIME_CONSTANT_S * elastic, plate_radius_m=PAD_RADIUS_M,
                reference_sinkage_m=REFERENCE_SINKAGE_M, metadata=json.dumps(dict(grid.metadata, note="SYNTHETIC soil; SCM linearisation of "
                                                                                    "lunar_regolith.contact.sinkage")))


def main():
    """Export terrain, soil maps and lander description for every site with soil columns."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sites", type=int, nargs="+", default=None)
    parser.add_argument("--half-size", type=float, default=12.0, help="terrain patch half size (m); the soil grids cover +-12 m")
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    arguments = parser.parse_args()
    sites = arguments.sites or sorted(int(p.stem.split("_")[1]) for p in (COLUMNS / "heterogeneous").glob("site_*.npz"))
    stage = open_scene()
    surface = TerrainGridSurface.from_mesh_prim(stage.GetPrimAtPath("/World/Terrain/Near2m"))
    lander = stage.GetPrimAtPath("/World/ApolloLM")
    from pxr import UsdPhysics

    mass_api = UsdPhysics.MassAPI(lander)
    pad_points, pad_centers = footpad_geometry(lander)
    all_sites = json.loads(study_paths.LANDER_NEAR_SITES.read_text())["points"]
    parameters = SinkageParameters(gravity_m_s2=GRAVITY, damping_time_constant_s=DAMPING_TIME_CONSTANT_S)
    for index in sites:
        site = all_sites[index]
        directory = OUTPUT / f"site_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        write_terrain_patch(surface, (site["x"], site["y"]), arguments.half_size, directory / "terrain.obj")
        for mode in ("heterogeneous", "uniform"):
            np.savez(directory / f"soil_{mode}.npz", **soil_maps(COLUMNS / mode / f"site_{index}.npz", parameters))
        yaw = math.radians(site["yaw_deg"])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        dish = np.vstack(list(pad_points.values())) @ rotation.T
        release_height = max(surface.height(site["x"] + p[0], site["y"] + p[1]) - p[2] for p in dish) + CLEARANCE_M
        releases = {f"{v:g}": dict(height=float(release_height), down_speed=math.sqrt(max(v ** 2 - 2 * GRAVITY * CLEARANCE_M, 0.0)))
                    for v in arguments.speeds}
        description = dict(
            site_index=index, x=site["x"], y=site["y"], yaw_deg=site["yaw_deg"], gravity=GRAVITY, clearance_m=CLEARANCE_M,
            mass_kg=float(mass_api.GetMassAttr().Get()), center_of_mass=list(map(float, mass_api.GetCenterOfMassAttr().Get())),
            diagonal_inertia=list(map(float, mass_api.GetDiagonalInertiaAttr().Get())), pad_radius_m=PAD_RADIUS_M,
            pad_bottom_centers={k: [float(v) for v in c] for k, c in pad_centers.items()}, releases=releases,
            terrain_patch=dict(file="terrain.obj", half_size_m=arguments.half_size, grid_m=surface.dx),
            note="Apollo LM (apollo_lander_modeling mass properties); footpads on ball joints; SYNTHETIC soil",
            provenance=study_paths.git_provenance())
        (directory / "lander.json").write_text(json.dumps(description, indent=1))
        maps = np.load(directory / "soil_heterogeneous.npz")
        print(f"site {index}: terrain patch, soil maps (Kphi {maps['bekker_kphi_pa_per_m'].min() / 1e6:.2f}-"
              f"{maps['bekker_kphi_pa_per_m'].max() / 1e6:.2f} MPa/m heterogeneous), lander.json -> {directory}")


if __name__ == "__main__":
    main()
