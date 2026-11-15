#!/usr/bin/env python3
"""Plate-penetration test in Project Chrono SCM vs lunar_regolith's pressure-sinkage model (PyChrono; Phase 2 cross-check, step 1).

    experiments/phase2_crosscheck/run_crosscheck.sh plate          (or: <CHRONO_PYTHON> chrono_plate_test.py)

A rigid disc the size of an Apollo LM footpad (radius 0.455 m) is driven into flat SCM soil at a constant slow speed to --depth, then
lifted.  The soil uses lunar_regolith's REFERENCE state as plain Bekker parameters (n = 1, k_c = 1.4 kN/m^2, k_phi = 820 kN/m^3, c = 0.17
kPa, phi = 35 deg, Janosi K = 1.8 cm), our unloading stiffness (10 x loading) as SCM's elastic stiffness, and no damping (slow test).
SCM's contact force on the disc / disc area = pressure.  For homogeneous soil our model is exactly Bekker's p = (k_c / b + k_phi) z, so
the loading branches should agree; SCM additionally discretises the soil on a grid (--delta) and treats the first part of each load
elastically.  Writes results/phase2/crosscheck/plate_test.json (both curves).  Uses numpy and pychrono only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pychrono.core as chrono
import pychrono.vehicle as vehicle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--radius", type=float, default=0.455)
parser.add_argument("--depth", type=float, default=0.10, help="maximum penetration (m)")
parser.add_argument("--speed", type=float, default=0.02, help="penetration speed (m/s)")
parser.add_argument("--delta", type=float, default=0.02, help="SCM grid spacing (m)")
parser.add_argument("--step", type=float, default=1e-3)
parser.add_argument("--out", type=Path, default=study_paths.RESULTS_ROOT / "phase2" / "crosscheck" / "plate_test.json")
arguments = parser.parse_args()

REFERENCE = dict(k_c=1400.0, k_phi=820000.0, n=1.0, cohesion=170.0, friction_deg=35.0, janosi_k=0.018)
UNLOADING_FACTOR = 10.0
plate_width = 2 * arguments.radius
loading_modulus = REFERENCE["k_c"] / plate_width + REFERENCE["k_phi"]          # our model, homogeneous reference soil (Pa/m)
area = math.pi * arguments.radius ** 2

system = chrono.ChSystemSMC()
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))                   # kinematic test: no gravity needed
plate = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, arguments.radius, 0.1, 1000.0, True, True, chrono.ChContactMaterialSMC())
plate.SetFixed(True)
system.Add(plate)
terrain = vehicle.SCMTerrain(system)
terrain.Initialize(3.0, 3.0, arguments.delta)
terrain.SetSoilParameters(REFERENCE["k_phi"], REFERENCE["k_c"], REFERENCE["n"], REFERENCE["cohesion"], REFERENCE["friction_deg"],
                          REFERENCE["janosi_k"], UNLOADING_FACTOR * loading_modulus, 0.0)
terrain.AddActiveDomain(plate, chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(2.2 * arguments.radius, 2.2 * arguments.radius, 0.5))

half_height = 0.05
records = []
duration_down = arguments.depth / arguments.speed
total = 2.0 * duration_down + 0.5
time_s = 0.0
force, torque = chrono.ChVector3d(), chrono.ChVector3d()
while time_s < total:
    if time_s <= duration_down:
        penetration, rate = arguments.speed * time_s, -arguments.speed
    else:
        penetration, rate = max(arguments.depth - arguments.speed * (time_s - duration_down), -0.02), arguments.speed
    plate.SetPos(chrono.ChVector3d(0, 0, half_height - penetration))
    plate.SetPosDt(chrono.ChVector3d(0, 0, rate))
    system.DoStepDynamics(arguments.step)
    time_s += arguments.step
    if terrain.GetContactForceBody(plate, force, torque):
        pressure = force.z / area
    else:
        pressure = 0.0
    if len(records) == 0 or time_s - records[-1]["t"] >= 0.05:
        records.append(dict(t=time_s, sinkage_m=penetration, chrono_pressure_pa=pressure, loading=time_s <= duration_down))

loading = [r for r in records if r["loading"] and r["sinkage_m"] > 0.01]
ratios = [r["chrono_pressure_pa"] / (loading_modulus * r["sinkage_m"]) for r in loading]
peak = max(r["chrono_pressure_pa"] for r in records)
unloaded = next((r["sinkage_m"] for r in records if not r["loading"] and r["chrono_pressure_pa"] <= 0.02 * peak), None)
expected_rebound = peak / (UNLOADING_FACTOR * loading_modulus)
summary = dict(
    our_loading_modulus_pa_per_m=loading_modulus, chrono_over_ours_loading_ratio_median=float(np.median(ratios)),
    chrono_over_ours_loading_ratio_range=[float(min(ratios)), float(max(ratios))], peak_pressure_pa=peak,
    pressure_released_at_sinkage_m=unloaded, our_model_releases_at_sinkage_m=arguments.depth - expected_rebound)
arguments.out.parent.mkdir(parents=True, exist_ok=True)
arguments.out.write_text(json.dumps(dict(summary=summary, reference_soil=REFERENCE, unloading_factor=UNLOADING_FACTOR, radius_m=arguments.radius,
                                         delta_m=arguments.delta, chrono_version=getattr(chrono, "__version__", "10"), records=records,
                                         provenance=study_paths.git_provenance()), indent=1))
print(json.dumps(summary, indent=1))
