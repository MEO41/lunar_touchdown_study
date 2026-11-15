#!/usr/bin/env python3
"""The Phase 2 deformable-soil touchdowns re-run in Project Chrono SCM (PyChrono; Phase 2 cross-check, step 2).

    experiments/phase2_crosscheck/run_crosscheck.sh touchdown          (or: <CHRONO_PYTHON> chrono_scm_touchdown.py [--sites ...])

Same landings as experiments/phase2_deformable/soil_touchdown.py (Isaac Sim + our per-footpad forces), in an independent engine:
  * terrain   SCM deformable grid (--delta) initialised from the Near2m patch (export_chrono_inputs.py); SCM deforms its own copy only.
              Chrono 10's SCM mesh initialisation needs the mesh near the origin and adds the mesh's lowest height again, so the whole
              scenario runs in a shifted frame: origin at the site, lowest patch point at z = 0 (all positions shifted alike).
              Sinkage is measured against SCM's OWN undeformed surface (GetInitHeight / GetInitNormal; within ~1-3 cm of the exact
              Near2m triangles because SCM samples the mesh on its grid).
  * soil      SCM's location-dependent parameters from soil_<mode>.npz: our layered pressure-sinkage linearised to Bekker Kphi (n = 1,
              Kc = 0) at the reference sinkage, our shallow cohesion / friction angle, Janosi K, our unloading stiffness as SCM's elastic
              stiffness and our damping.  (SCM is single-layer Bekker per grid node; layering enters only through that linearisation.)
  * lander    one rigid body with the Apollo LM's mass, centre of mass and inertia (minus the pads), and four footpad discs (radius 0.455
              m, 50 kg; total mass unchanged) on spherical joints at the pad centre - the ball-jointed pads of the real LM.  Only the pads touch the soil.
  * release   identical height, yaw and downward speed as in Isaac (lander.json).
Recorded: per-pad sinkage (pad-base centre below the UNDEFORMED surface, along its normal - the same definition as in Isaac), SCM
contact force per pad (final shares averaged over the last second), tilt, rest time, outcome.  Writes
results/phase2/crosscheck/chrono_<mode>_v<speed>.json.  numpy + pychrono only.  Soil is SYNTHETIC.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pychrono.core as chrono
import pychrono.vehicle as vehicle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402
from landing_geometry import TerrainGridSurface, plane_slope_deg, support_margin_m  # noqa: E402

INPUTS = study_paths.RESULTS_ROOT / "phase2" / "chrono_inputs"
OUTPUT = study_paths.phase_results_directory("phase2", "crosscheck")
PADS = ("+X", "+Y", "-X", "-Y")
PAD_THICKNESS_M = 0.1
SETTINGS = dict(pad_mass_kg=50.0, damping_scale=1.0)
REST_WINDOW_S, REST_EARLIEST_S, REST_DRIFT_M, REST_TURN_DEG = 3.0, 5.0, 0.005, 0.05     # as soil_touchdown.py
RECORD_S = 1.0 / 60.0


def load_surface(obj_path):
    """The undeformed terrain patch (regular 2 m grid in the OBJ) as a TerrainGridSurface."""
    vertices = np.array([list(map(float, line.split()[1:4])) for line in obj_path.read_text().splitlines() if line.startswith("v ")])
    xs, ys = np.unique(vertices[:, 0]), np.unique(vertices[:, 1])
    heights = np.full((ys.size, xs.size), np.nan)
    heights[np.searchsorted(ys, vertices[:, 1]), np.searchsorted(xs, vertices[:, 0])] = vertices[:, 2]
    grid = dict(x0=xs[0], y0=ys[0], dx=xs[1] - xs[0], dy=ys[1] - ys[0], nx=xs.size, ny=ys.size)
    return TerrainGridSurface(grid, heights)


class SoilMaps(vehicle.SoilParametersCallback):
    """SCM soil parameters at any location: bilinear in the exported maps (outside: nearest edge)."""

    def __init__(self, maps):
        """Keep the map arrays."""
        vehicle.SoilParametersCallback.__init__(self)
        self.east, self.north = maps["east"], maps["north"]
        self.fields = {k: maps[k] for k in ("bekker_kphi_pa_per_m", "cohesion_pa", "friction_angle_deg", "elastic_k_pa_per_m", "damping_pa_s_per_m")}
        self.janosi = float(maps["janosi_k_m"])

    def value(self, name, x, y):
        """Bilinear value of one map at (x, y)."""
        fe = np.interp(x, self.east, np.arange(self.east.size))
        fn = np.interp(y, self.north, np.arange(self.north.size))
        j, i = min(int(fe), self.east.size - 2), min(int(fn), self.north.size - 2)
        u, v = fe - j, fn - i
        a = self.fields[name]
        return float((1 - u) * (1 - v) * a[i, j] + u * (1 - v) * a[i, j + 1] + (1 - u) * v * a[i + 1, j] + u * v * a[i + 1, j + 1])

    def Set(self, location, kphi, kc, n, cohesion, friction, janosi, elastic, damping):
        """SCM callback: write the parameters for one grid node."""
        x, y = location.x, location.y
        vehicle.doublep_assign(kphi, self.value("bekker_kphi_pa_per_m", x, y))
        vehicle.doublep_assign(kc, 0.0)
        vehicle.doublep_assign(n, 1.0)
        vehicle.doublep_assign(cohesion, self.value("cohesion_pa", x, y))
        vehicle.doublep_assign(friction, self.value("friction_angle_deg", x, y))
        vehicle.doublep_assign(janosi, self.janosi)
        vehicle.doublep_assign(elastic, self.value("elastic_k_pa_per_m", x, y))
        vehicle.doublep_assign(damping, SETTINGS["damping_scale"] * self.value("damping_pa_s_per_m", x, y))


def shifted_mesh(obj_path, offset, out_path):
    """Copy of the terrain OBJ moved by -offset (comments dropped: Chrono's OBJ reader mis-computes the bounding box with them)."""
    lines = []
    for line in obj_path.read_text().splitlines():
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            lines.append(f"v {x - offset[0]:.4f} {y - offset[1]:.4f} {z - offset[2]:.4f}")
        elif not line.startswith("#"):
            lines.append(line)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def vector(v):
    """chrono vector -> numpy."""
    return np.array([v.x, v.y, v.z])


def run_landing(site_directory, mode, speed, delta, step, max_seconds):
    """One touchdown; returns the result record (same fields as soil_touchdown.py where they exist)."""
    description = json.loads((site_directory / "lander.json").read_text())
    maps = dict(np.load(site_directory / f"soil_{mode}.npz"))
    surface = load_surface(site_directory / "terrain.obj")
    release = description["releases"][f"{speed:g}"]
    gravity = description["gravity"]

    offset = np.array([description["x"], description["y"], float(np.nanmin(surface.heights))])
    mesh = shifted_mesh(site_directory / "terrain.obj", offset, site_directory / "terrain_chrono_frame.obj")
    system = chrono.ChSystemSMC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -gravity))
    terrain = vehicle.SCMTerrain(system)
    terrain.Initialize(str(mesh), delta)
    maps["east"], maps["north"] = maps["east"] - offset[0], maps["north"] - offset[1]
    soil = SoilMaps(maps)
    terrain.RegisterSoilParametersCallback(soil)

    yaw = math.radians(description["yaw_deg"])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    orientation = chrono.QuatFromAngleZ(yaw)
    origin = np.array([description["x"], description["y"], release["height"]]) - offset
    com_local = np.array(description["center_of_mass"])
    down = chrono.ChVector3d(0, 0, -release["down_speed"])

    body = chrono.ChBody()
    body.SetMass(description["mass_kg"] - 4 * SETTINGS["pad_mass_kg"])
    body.SetInertiaXX(chrono.ChVector3d(*description["diagonal_inertia"]))
    body.SetPos(chrono.ChVector3d(*(origin + rotation @ com_local)))
    body.SetRot(orientation)
    body.SetPosDt(down)
    system.Add(body)
    pads = {}
    pad_volume = math.pi * description["pad_radius_m"] ** 2 * PAD_THICKNESS_M
    for k in PADS:
        bottom = origin + rotation @ np.array(description["pad_bottom_centers"][k])
        pad = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, description["pad_radius_m"], PAD_THICKNESS_M, SETTINGS["pad_mass_kg"] / pad_volume, True, True,
                                        chrono.ChContactMaterialSMC())
        pad.SetPos(chrono.ChVector3d(*(bottom + [0, 0, PAD_THICKNESS_M / 2])))
        pad.SetRot(orientation)
        pad.SetPosDt(down)
        system.Add(pad)
        joint = chrono.ChLinkLockSpherical()
        joint.Initialize(pad, body, chrono.ChFramed(chrono.ChVector3d(*(bottom + [0, 0, PAD_THICKNESS_M])), chrono.QUNIT))
        system.Add(joint)
        terrain.AddActiveDomain(pad, chrono.ChVector3d(0, 0, 0),
                                chrono.ChVector3d(2.4 * description["pad_radius_m"], 2.4 * description["pad_radius_m"], 0.6))
        pads[k] = pad

    force, torque = chrono.ChVector3d(), chrono.ChVector3d()
    time_s, next_record = 0.0, 0.0
    history, series, recent = [], [], []
    max_sinkage = {k: 0.0 for k in PADS}
    peak_force = {k: 0.0 for k in PADS}
    peak_tilt, rest_time = 0.0, None
    wall = time.time()
    while time_s < max_seconds and rest_time is None:
        system.DoStepDynamics(step)
        time_s += step
        sinkage, vertical = {}, {}
        for k, pad in pads.items():
            bottom_point = pad.TransformPointLocalToParent(chrono.ChVector3d(0, 0, -PAD_THICKNESS_M / 2))
            bottom = vector(bottom_point)
            sinkage[k] = (terrain.GetInitHeight(bottom_point) - bottom[2]) * terrain.GetInitNormal(bottom_point).z
            vertical[k] = vector(force)[2] if terrain.GetContactForceBody(pad, force, torque) else 0.0
            max_sinkage[k] = max(max_sinkage[k], sinkage[k])
            peak_force[k] = max(peak_force[k], vertical[k])
        recent.append(vertical)
        if len(recent) > int(1.0 / step):
            recent.pop(0)
        up = vector(body.GetRot().GetAxisZ())
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        peak_tilt = max(peak_tilt, tilt)
        if not np.isfinite(vector(body.GetPos())).all():
            raise FloatingPointError(f"simulation diverged at t = {time_s:.3f} s (try a smaller --step or larger --pad-mass)")
        if time_s >= next_record:
            next_record += RECORD_S
            total = sum(max(v, 0.0) for v in vertical.values())
            series.append([round(time_s, 4), round(tilt, 3)] + [round(sinkage[k], 4) for k in PADS]
                          + [round(max(vertical[k], 0.0) / total, 4) if total > 0 else 0.0 for k in PADS])
            history.append((vector(body.GetPos()), up))
            window = history[-int(REST_WINDOW_S / RECORD_S):]
            if time_s > REST_EARLIEST_S and len(window) > 10:
                positions = np.array([h[0] for h in window])
                ups = np.array([h[1] for h in window])
                drift = np.max(np.linalg.norm(positions - positions[-1], axis=1))
                turn = np.degrees(np.max(np.arccos(np.clip(ups @ ups[-1], -1, 1))))
                if drift < REST_DRIFT_M and turn < REST_TURN_DEG:
                    rest_time = time_s
    after_landing = max(0, int(REST_EARLIEST_S / RECORD_S))           # creep only after the touchdown transient (t >= 5 s), last 5 s at most
    tail = history[after_landing:][-int(5.0 / RECORD_S):]
    creep_speed = float(np.linalg.norm(tail[-1][0][:2] - tail[0][0][:2]) / max(len(tail) * RECORD_S, 1e-9)) if len(tail) > 1 else None
    tilt_drift = float(np.degrees(np.arccos(np.clip(np.dot(tail[-1][1], tail[0][1]), -1, 1))) / max(len(tail) * RECORD_S, 1e-9)) if len(tail) > 1 else None
    mean_vertical = {k: max(float(np.mean([r[k] for r in recent])), 0.0) for k in PADS}
    total = sum(mean_vertical.values())
    weight = description["mass_kg"] * gravity
    touching = [k for k in PADS if mean_vertical[k] > 0.002 * weight]
    centers = {k: vector(pads[k].GetPos()) for k in PADS}
    surface_gap = float(np.mean([terrain.GetInitHeight(chrono.ChVector3d(*c[:2], 0)) - (surface.height(*(c[:2] + offset[:2])) - offset[2])
                                 for c in centers.values()]))
    com = vector(body.GetPos())
    margin = support_margin_m([centers[k][:2] for k in touching], com[:2]) if len(touching) >= 3 else float("nan")
    slope = plane_slope_deg(surface, description["x"], description["y"])
    if tilt > 45:
        outcome = "toppled"
    elif rest_time is None:
        outcome = "moving"
    else:
        outcome = "upright" if len(touching) >= 3 and margin == margin and margin > 0 and tilt <= slope + 5 else "tilted"
    return dict(
        idx=description["site_index"], x=description["x"], y=description["y"], yaw_deg=description["yaw_deg"], slope_deg=slope,
        touchdown_speed_target=speed, soil_mode=mode, tilt_deg=tilt, peak_tilt_deg=peak_tilt, rest_s=rest_time, touching=touching,
        com_margin_m=margin, outcome=outcome, weight_N=weight, step_s=step, creep_speed_last_5s_m_per_s=creep_speed,
        tilt_drift_last_5s_deg_per_s=tilt_drift, scm_minus_exact_surface_at_pads_m=surface_gap, mean_vertical_force_sum_N=total, wall_s=time.time() - wall,
        pads={k: dict(max_sinkage_m=max_sinkage[k], final_sinkage_m=sinkage[k], peak_normal_force_N=peak_force[k],
                      final_load_share=mean_vertical[k] / total if total > 0 else 0.0) for k in PADS},
        series_columns=["t_s", "tilt_deg", *[f"sinkage_{k}_m" for k in PADS], *[f"share_{k}" for k in PADS]], series=series)


def main():
    """All sites x speeds x soil modes (or a subset); one JSON per mode and speed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sites", type=int, nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=["heterogeneous", "uniform"])
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--delta", type=float, default=0.1, help="SCM grid spacing (m); 0.05 gave the same pad sinkage within 0.4 cm at site 20")
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--pad-mass", type=float, default=50.0, help="footpad disc mass (kg); 5 kg discs on ball joints diverge at 1 ms steps under the stiff soil damping")
    parser.add_argument("--damping-scale", type=float, default=1.0, help="multiplier on the exported soil damping")
    arguments = parser.parse_args()
    SETTINGS.update(pad_mass_kg=arguments.pad_mass, damping_scale=arguments.damping_scale)
    sites = arguments.sites or sorted(int(p.name.split("_")[1]) for p in INPUTS.glob("site_*"))
    for mode in arguments.modes:
        for speed in arguments.speeds:
            results = []
            for index in sites:
                step = arguments.step
                while True:
                    try:
                        record = run_landing(INPUTS / f"site_{index}", mode, speed, arguments.delta, step, arguments.max_seconds)
                        break
                    except FloatingPointError as error:            # very firm soil: stiff SCM elastic / damping terms need a finer step
                        if step < arguments.step / 16:
                            raise
                        print(f"{mode} {speed:g} m/s site {index}: {error}; retrying with step {step / 4:g} s", flush=True)
                        step /= 4
                results.append(record)
                print(f"{mode} {speed:g} m/s site {index}: tilt {record['tilt_deg']:.2f} (peak {record['peak_tilt_deg']:.2f}), rest "
                      f"{record['rest_s']}, max sinkage cm " + ", ".join(f"{k} {100 * p['max_sinkage_m']:.1f}" for k, p in record["pads"].items())
                      + f", {record['outcome']}, creep {1000 * (record['creep_speed_last_5s_m_per_s'] or 0):.2f} mm/s "
                      f"(step {record['step_s']:g} s, {record['wall_s']:.0f} s wall)", flush=True)
            path = OUTPUT / f"chrono_{mode}_v{speed:g}.json"
            path.write_text(json.dumps(dict(
                args=dict(mode=mode, speed=speed, delta_m=arguments.delta, step_s=arguments.step, pad_mass_kg=SETTINGS["pad_mass_kg"],
                          damping_scale=SETTINGS["damping_scale"],
                          note="Project Chrono SCM cross-check of soil_touchdown.py; SYNTHETIC soil (lunar_regolith) linearised to Bekker",
                          provenance=study_paths.git_provenance()),
                results=results)))
            print("saved", path, flush=True)


if __name__ == "__main__":
    main()
