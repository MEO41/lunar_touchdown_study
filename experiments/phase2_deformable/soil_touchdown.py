#!/usr/bin/env python3
"""Apollo LM touchdown on DEFORMABLE regolith: per-footpad pressure-sinkage and shear from lunar_regolith (Isaac Sim Python).  Phase 2.

    experiments/phase2_deformable/run_phase2.sh soil-touchdown [speed]
    <lander repo>/isaac_run.sh soil_touchdown.py --soil-mode heterogeneous --touchdown-speed 1 [--sites 56 200 ...]

Coupling (lander side; the soil law itself is general and lives in lunar_regolith.contact.sinkage):
  * The footpads and the lowest leg parts (every lander collider lower than LOW_PART_HEIGHT_M and within PAD_REACH_M of a pad) are
    filtered out of collisions with the terrain (PhysX collision groups).  Everything else - body, engine bell, upper legs - still
    collides rigidly with the terrain mesh, so a lander that tips over still lands on its side.
  * The Apollo LM footpads sit on ball joints: a pad turns to lie on the soil and transmits no moment.  So each pad is a disc parallel
    to the local terrain, whose sinkage is the depth of the pad-base centre below the UNDEFORMED Near2m surface (LROC NAC DTM, never
    changed), measured along the terrain normal before every physics step (240 Hz).  The soil under the disc is sampled at 19
    area-weighted points (lunar_regolith CircularPatch, 2 rings), each with its own lunar_regolith PlateContact (the plate's
    pressure-sinkage curve; permanent sinkage; Janosi shear), so a pad half over a soft pocket feels half of it.  The resulting force
    acts at the pad centre; the four pad forces are summed into one force and torque on the lander (omni.physics.tensors).
  * The soil under each pad comes from a SoilColumnGrid (lunar_regolith contact-columns) around the site: SYNTHETIC, uncalibrated
    soil fields; the pressure-sinkage constants are lunar literature values (see sinkage.py), their scaling with the soil fields is
    an assumption.
Recorded per landing: per-pad sinkage (max and final), normal force history and final share, the soil stiffness under each pad,
tilt history, support margin, outcome class, and (with --record-poses) poses for rendering.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402
from landing_geometry import TerrainGridSurface, footpad_geometry, plane_slope_deg, support_margin_m  # noqa: E402

sys.path.insert(0, str(study_paths.REGOLITH_REPOSITORY / "src"))
from lunar_regolith.contact.patches import CircularPatch  # noqa: E402
from lunar_regolith.contact.sinkage import PlateContact, PressureSinkageField, SinkageParameters, SoilColumnGrid  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=study_paths.LANDER_SCENE)
parser.add_argument("--lander", type=Path, default=study_paths.LANDER_USD)
parser.add_argument("--points", type=Path, default=study_paths.LANDER_NEAR_SITES)
parser.add_argument("--soil-mode", default="heterogeneous", help="sub-directory of --columns-directory: heterogeneous | uniform")
parser.add_argument("--columns-directory", type=Path, default=study_paths.RESULTS_ROOT / "phase2" / "soil_columns",
                    help="<dir>/<soil mode>/site_<index>.npz: one SoilColumnGrid per site (prepare_soil_columns.py)")
parser.add_argument("--sites", type=int, nargs="+", default=None, help="site indices (default: every site with a column grid)")
parser.add_argument("--touchdown-speed", type=float, default=1.0, help="vertical speed at first contact (m/s)")
parser.add_argument("--clearance", type=float, default=0.25)
parser.add_argument("--pad-radius", type=float, default=0.455, help="footpad plate radius (m); Apollo LM footpad 0.91 m across")
parser.add_argument("--physics-hz", type=float, default=480.0,
                    help="physics rate; with --damping-time-constant it must keep the explicit coupling stable: stiffness x step^2 / pad mass "
                         "and damping x step / pad mass well below 1 (480 Hz and 0.003 s: < 0.6 up to ~20 MPa/m soil)")
parser.add_argument("--damping-time-constant", type=float, default=0.003, help="soil damping = this x current stiffness (s)")
parser.add_argument("--friction", type=float, nargs=2, default=(0.8, 0.7), help="terrain friction for the parts that still collide")
parser.add_argument("--max-seconds", type=float, default=20.0)
parser.add_argument("--record-poses", action="store_true")
parser.add_argument("--out", type=Path, default=None)
arguments = parser.parse_args()
columns_directory = arguments.columns_directory / arguments.soil_mode
if arguments.sites is None:
    arguments.sites = sorted(int(p.stem.split("_")[1]) for p in columns_directory.glob("site_*.npz"))
if arguments.out is None:
    arguments.out = study_paths.phase_results_directory("phase2", "soil_touchdown") / f"soil_touchdown_{arguments.soil_mode}_v{arguments.touchdown_speed:g}.json"

from isaacsim import SimulationApp  # noqa: E402

simulation_application = SimulationApp({"headless": True})

import omni.physics.tensors as physics_tensors  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

PADS = ("+X", "+Y", "-X", "-Y")
TERRAIN_MESH_PATH = "/World/Terrain/Near2m"
LOW_PART_HEIGHT_M = 0.6          # lander colliders reaching lower than this (lander frame) ...
PAD_REACH_M = 1.2                # ... within this horizontal distance of a pad centre do not touch the terrain: the soil model carries them
RECORD_EVERY_STEPS = 4           # series at physics_hz / 4
REST_WINDOW_S, REST_EARLIEST_S, REST_DRIFT_M, REST_TURN_DEG = 3.0, 5.0, 0.005, 0.05   # at rest: moved < 5 mm and turned < 0.05 deg in 3 s


def update(count=1):
    """Advance the application `count` frames."""
    for _ in range(count):
        simulation_application.update()


# ------------------------------------------------------------------------------------------------------------- stage
context = omni.usd.get_context()
context.open_stage(str(arguments.scene))
update(20)
while context.get_stage_loading_status()[2] > 0:
    update()
stage = context.get_stage()
world_variant_sets = stage.GetPrimAtPath("/World").GetVariantSets()
world_variant_sets.GetVariantSet("terrain").SetVariantSelection("nac_2m")
world_variant_sets.GetVariantSet("regolith").SetVariantSelection("rigid")
update(20)
session_layer = stage.GetSessionLayer()
stage.SetEditTarget(Usd.EditTarget(session_layer))
for physics_scene in [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]:
    PhysxSchema.PhysxSceneAPI.Apply(physics_scene).CreateTimeStepsPerSecondAttr(int(arguments.physics_hz))
GRAVITY = 1.62
regolith_material = stage.GetPrimAtPath("/World/Terrain/PhysicsMaterials/Regolith")
UsdPhysics.MaterialAPI(regolith_material).GetStaticFrictionAttr().Set(arguments.friction[0])
UsdPhysics.MaterialAPI(regolith_material).GetDynamicFrictionAttr().Set(arguments.friction[1])
regolith_material.AddAppliedSchema("PhysxMaterialAPI")
regolith_material.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("multiply")
footpad_material = UsdShade.Material.Define(stage, "/World/LanderContactMaterial")
footpad_material_api = UsdPhysics.MaterialAPI.Apply(footpad_material.GetPrim())
footpad_material_api.CreateStaticFrictionAttr(1.0)
footpad_material_api.CreateDynamicFrictionAttr(1.0)
footpad_material_api.CreateRestitutionAttr(0.0)

surface = TerrainGridSurface.from_mesh_prim(stage.GetPrimAtPath(TERRAIN_MESH_PATH))

# ------------------------------------------------------------------------------------------------------------- lander
scene_lander = stage.GetPrimAtPath("/World/ApolloLM")
mass_api = UsdPhysics.MassAPI(scene_lander)
MASS = mass_api.GetMassAttr().Get()
CENTER_OF_MASS = mass_api.GetCenterOfMassAttr().Get()
INERTIA = mass_api.GetDiagonalInertiaAttr().Get()
center_of_mass_local = np.array(CENTER_OF_MASS, np.float64)
pad_dish_points, pad_bottom_center = footpad_geometry(scene_lander)
scene_lander.SetActive(False)

lander_file = Usd.Stage.Open(str(arguments.lander))
lander_root = lander_file.GetDefaultPrim()
bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
soil_carried_parts = []
for prim in Usd.PrimRange(lander_root):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
    low, high = np.array(box.GetMin()), np.array(box.GetMax())
    near_pad = any(np.hypot(*(np.clip(c[:2], low[:2], high[:2]) - c[:2])) < PAD_REACH_M for c in pad_bottom_center.values())
    if low[2] < LOW_PART_HEIGHT_M and near_pad:
        soil_carried_parts.append(str(prim.GetPath())[len(str(lander_root.GetPath())):])
print(f"{len(soil_carried_parts)} lander colliders near the pads are carried by the soil model, not by terrain collisions", flush=True)

parameters = SinkageParameters(gravity_m_s2=GRAVITY, damping_time_constant_s=arguments.damping_time_constant)
soil_by_site, grid_metadata = {}, {}
for index in arguments.sites:
    grid = SoilColumnGrid.load(columns_directory / f"site_{index}.npz")
    soil_by_site[index] = PressureSinkageField(grid, arguments.pad_radius, parameters)
    grid_metadata[index] = grid.metadata
PAD_AREA = math.pi * arguments.pad_radius ** 2
patch_east, patch_north, patch_weight = CircularPatch(arguments.pad_radius, rings=2).sample_offsets()

sites = json.loads(arguments.points.read_text())["points"]
for index, site in enumerate(sites):
    site["site_index"] = index
sites = [sites[i] for i in arguments.sites]
for site in sites:
    grid = soil_by_site[site["site_index"]].grid
    if not (grid.contains(site["x"] - 8, site["y"] - 8) and grid.contains(site["x"] + 8, site["y"] + 8)):
        raise SystemExit(f"site {site['site_index']} is not covered by its soil column grid")

# ------------------------------------------------------------------------------------------------------------- landers
UsdGeom.Xform.Define(stage, "/World/Drops")
terrain_group = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/Terrain")
terrain_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Terrain")
soil_group = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/SoilCarriedLanderParts")
soil_group.CreateFilteredGroupsRel().AddTarget(terrain_group.GetPath())
soil_includes = soil_group.GetCollidersCollectionAPI().CreateIncludesRel()
landers = []
for number, site in enumerate(sites):
    path = f"/World/Drops/L{number:04d}"
    xform = UsdGeom.Xform.Define(stage, path)
    xform.GetPrim().GetReferences().AddReference(str(arguments.lander))
    yaw = math.radians(site["yaw_deg"])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    all_dish = np.vstack(list(pad_dish_points.values())) @ rotation.T
    release_height = max(surface.height(site["x"] + p[0], site["y"] + p[1]) - p[2] for p in all_dish) + arguments.clearance
    initial_down_speed = math.sqrt(max(arguments.touchdown_speed ** 2 - 2 * GRAVITY * arguments.clearance, 0.0))
    xform.AddTranslateOp().Set(Gf.Vec3d(site["x"], site["y"], release_height))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass.CreateMassAttr(float(MASS))
    mass.CreateCenterOfMassAttr(CENTER_OF_MASS)
    mass.CreateDiagonalInertiaAttr(INERTIA)
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    UsdPhysics.RigidBodyAPI(xform.GetPrim()).CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, -float(initial_down_speed)))
    PhysxSchema.PhysxContactReportAPI.Apply(xform.GetPrim()).CreateThresholdAttr(0.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim()).CreateSleepThresholdAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(xform.GetPrim()).Bind(footpad_material, UsdShade.Tokens.strongerThanDescendants, "physics")
    for part in soil_carried_parts:
        soil_includes.AddTarget(Sdf.Path(path + part))
    landers.append(dict(site=site, path=path, release_height=float(release_height),
                        soil=soil_by_site[site["site_index"]],
                        contacts={k: [PlateContact(soil_by_site[site["site_index"]]) for _ in patch_weight] for k in PADS}, body_contact_seen=False,
                        series=[], pose_series=[], peak_tilt=0.0, min_support_margin=float("inf"),
                        max_sinkage={k: 0.0 for k in PADS}, peak_force={k: 0.0 for k in PADS}, first_contact={k: None for k in PADS},
                        last_force={k: 0.0 for k in PADS}, history=[], rest_time=None))
update(5)

# body contacts (anything that still collides with the terrain: should stay empty in a normal landing)
body_contact_paths = set()
body_contact_parts = {}


def on_contact_report(headers, data):
    """Note which landers touch the terrain with a rigid (non-soil-carried) part."""
    for header in headers:
        path0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
        path1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
        for lander_path, other in ((path0, path1), (path1, path0)):
            carried = sum(abs(data[i].impulse.x) + abs(data[i].impulse.y) + abs(data[i].impulse.z)
                          for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if lander_path.startswith("/World/Drops/L") and "/Terrain/" in other and carried > 1e-3:
                lander_root_path = "/".join(lander_path.split("/")[:4])
                body_contact_paths.add(lander_root_path)
                body_contact_parts.setdefault(lander_root_path, set()).add(lander_path[len(lander_root_path):])


contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)

# ------------------------------------------------------------------------------------------------------------- soil forces
state = dict(view=None, step=0, time_s=0.0)
step_seconds = 1.0 / arguments.physics_hz


def quaternion_matrix(qx, qy, qz, qw):
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def pad_soil_forces(lander, position, rotation, linear_velocity, angular_velocity, center_of_mass):
    """Soil force on each pad (ball joint: pad parallel to the terrain, force at the pad centre, no pad moment):
    {pad: (force vector, torque about the COM, centre point, sinkage, normal force)}."""
    result = {}
    for k in PADS:
        center = rotation @ pad_bottom_center[k] + position
        normal = surface.normal(center[0], center[1])
        sinkage = (surface.height(center[0], center[1]) - center[2]) * normal[2]
        velocity = linear_velocity + np.cross(angular_velocity, center - center_of_mass)
        sinkage_rate = -float(np.dot(velocity, normal))
        tangent_east = np.cross([0.0, 1.0, 0.0], normal)
        tangent_east /= np.linalg.norm(tangent_east)
        tangent_north = np.cross(normal, tangent_east)
        pressure, shear = 0.0, np.zeros(2)
        for east_offset, north_offset, weight, contact in zip(patch_east, patch_north, patch_weight, lander["contacts"][k]):
            sample = center + east_offset * tangent_east + north_offset * tangent_north
            stress = contact.stress(sample[0], sample[1], sinkage, sinkage_rate, sample[:2])
            pressure += weight * stress.pressure_pa
            shear += weight * stress.shear_pa
        shear_vector = np.array([shear[0], shear[1], 0.0])
        shear_vector[2] = -(normal[0] * shear_vector[0] + normal[1] * shear_vector[1]) / normal[2]   # into the tangent plane
        force = PAD_AREA * (pressure * normal + shear_vector)
        result[k] = (force, np.cross(center - center_of_mass, force), center, float(sinkage), PAD_AREA * pressure)
    return result


def on_physics_step(dt):
    """Pre-step: measure every pad, apply the soil forces, record."""
    view = state["view"]
    if view is None:
        return
    transforms = view.get_transforms().reshape(-1, 7)
    velocities = view.get_velocities().reshape(-1, 6)
    forces = np.zeros((len(landers), 3), np.float32)
    torques = np.zeros((len(landers), 3), np.float32)
    positions = np.zeros((len(landers), 3), np.float32)
    record = state["step"] % RECORD_EVERY_STEPS == 0
    for index, lander in enumerate(landers):
        position = transforms[index, :3].astype(np.float64)
        rotation = quaternion_matrix(*transforms[index, 3:7].astype(np.float64))
        center_of_mass = rotation @ center_of_mass_local + position
        pads = pad_soil_forces(lander, position, rotation, velocities[index, :3].astype(np.float64),
                               velocities[index, 3:].astype(np.float64), center_of_mass)
        total_force, total_torque = np.zeros(3), np.zeros(3)
        for k, (force, torque, point, sinkage, normal_force) in pads.items():
            total_force += force
            total_torque += torque
            lander["max_sinkage"][k] = max(lander["max_sinkage"][k], sinkage)
            lander["peak_force"][k] = max(lander["peak_force"][k], normal_force)
            lander["last_force"][k] = normal_force
            if normal_force > 0 and lander["first_contact"][k] is None:
                lander["first_contact"][k] = state["time_s"]
        forces[index], torques[index], positions[index] = total_force, total_torque, center_of_mass
        recent = lander.setdefault("recent_pad_forces", [])          # last second, every step (time-averaged final loads)
        recent.append(np.array([pads[k][0] for k in PADS]))
        if len(recent) > int(arguments.physics_hz):
            recent.pop(0)
        lander["recent_center_of_mass"] = center_of_mass
        up = rotation[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        lander["peak_tilt"] = max(lander["peak_tilt"], tilt)
        loaded = [k for k in PADS if lander["last_force"][k] > 0.002 * MASS * GRAVITY]
        if len(loaded) >= 3:
            margin = support_margin_m([pads[k][2][:2] for k in loaded], center_of_mass[:2])
            if math.isfinite(margin):
                lander["min_support_margin"] = min(lander["min_support_margin"], margin)
        if record:
            total_normal = sum(lander["last_force"].values())
            lander["series"].append([round(state["time_s"], 4), round(tilt, 3)]
                                    + [round(pads[k][3], 4) for k in PADS]
                                    + [round(lander["last_force"][k] / total_normal, 4) if total_normal > 0 else 0.0 for k in PADS])
            if arguments.record_poses:
                qx, qy, qz, qw = transforms[index, 3:7].astype(np.float64)
                lander["pose_series"].append([round(state["time_s"], 4), *[round(float(v), 5) for v in position],
                                              round(float(qw), 6), round(float(qx), 6), round(float(qy), 6), round(float(qz), 6)])
            lander["history"].append((position.copy(), up.copy()))
            lander["pads_now"] = {k: (pads[k][2], pads[k][3]) for k in PADS}
            lander["pad_forces_now"] = {k: pads[k][0] for k in PADS}
            lander["center_of_mass_now"] = center_of_mass
    view.apply_forces_and_torques_at_position(forces, torques, positions, np.arange(len(landers), dtype=np.uint32), True)
    state["step"] += 1
    state["time_s"] += dt


physics_interface = omni.physx.get_physx_interface()
step_subscription = physics_interface.subscribe_physics_on_step_events(on_physics_step, True, 0)

# ------------------------------------------------------------------------------------------------------------- run
timeline = omni.timeline.get_timeline_interface()
timeline.play()
update(2)
stage_id = context.get_stage_id()
if not omni.physx.get_physx_simulation_interface().is_stage_attached() if hasattr(omni.physx.get_physx_simulation_interface(), "is_stage_attached") else False:
    omni.physx.get_physx_simulation_interface().attach_stage(stage_id)
simulation_view = physics_tensors.create_simulation_view("numpy", stage_id=stage_id)
simulation_view.set_subspace_roots("/")
state["view"] = simulation_view.create_rigid_body_view("/World/Drops/L*")
if state["view"].count != len(landers):
    raise SystemExit(f"rigid body view found {state['view'].count} landers, expected {len(landers)}")
wall_start = time.time()
while state["time_s"] < arguments.max_seconds and any(l["rest_time"] is None for l in landers):
    update(1)
    for lander in landers:
        history = lander["history"][-int(REST_WINDOW_S * arguments.physics_hz / RECORD_EVERY_STEPS):]
        if lander["rest_time"] is None and state["time_s"] > REST_EARLIEST_S and len(history) > 10:
            positions = np.array([h[0] for h in history])
            ups = np.array([h[1] for h in history])
            drift = np.max(np.linalg.norm(positions - positions[-1], axis=1))
            turn = np.degrees(np.max(np.arccos(np.clip(ups @ ups[-1], -1, 1))))
            if drift < REST_DRIFT_M and turn < REST_TURN_DEG:
                lander["rest_time"] = state["time_s"]
timeline.stop()
print(f"simulated {state['time_s']:.1f} s at {arguments.physics_hz:g} Hz in {time.time() - wall_start:.0f} s wall", flush=True)


def outcome(tilt, rest_time, touching, margin, slope, tolerance_deg=5.0):
    """Outcome class (as in plot_drop_test.py)."""
    if tilt > 45:
        return "toppled"
    if rest_time is None:
        return "moving"
    return "upright" if len(touching) >= 3 and margin == margin and margin > 0 and tilt <= slope + tolerance_deg else "tilted"


results = []
for lander in landers:
    site = lander["site"]
    position, up = lander["history"][-1]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    # final loads: averaged over the last second (a lightly loaded pad sitting at the bottom of its own dent chatters in and out of
    # contact from step to step, so a single step can miss it)
    mean_forces = np.mean(lander["recent_pad_forces"], axis=0)
    mean_vertical = {k: max(float(mean_forces[PADS.index(k)][2]), 0.0) for k in PADS}
    total = sum(mean_vertical.values())
    touching = [k for k in PADS if mean_vertical[k] > 0.002 * MASS * GRAVITY]
    final_points = lander.get("pads_now", {})
    slope = plane_slope_deg(surface, site["x"], site["y"])
    margin = (support_margin_m([final_points[k][0][:2] for k in touching], lander["center_of_mass_now"][:2])
              if len(touching) >= 3 else float("nan"))
    pads = {}
    for k in PADS:
        point = final_points.get(k, (np.zeros(3), 0.0))[0]
        pads[k] = dict(max_sinkage_m=lander["max_sinkage"][k], final_sinkage_m=final_points.get(k, (None, 0.0))[1],
                       peak_normal_force_N=lander["peak_force"][k], final_load_share=mean_vertical[k] / total if total > 0 else 0.0,
                       first_contact_s=lander["first_contact"][k], final_point=[float(v) for v in point],
                       soil_tangent_modulus_at_5cm_Pa_per_m=lander["soil"].tangent_modulus(float(point[0]), float(point[1]), 0.05),
                       deepest_plastic_sinkage_m=max(c.deepest_sinkage_m for c in lander["contacts"][k]),
                       final_force_N=[float(v) for v in lander.get("pad_forces_now", {}).get(k, np.zeros(3))],
                       mean_force_last_second_N=[float(v) for v in np.mean(lander["recent_pad_forces"], axis=0)[PADS.index(k)]],
                       loaded_fraction_of_steps_last_second=float(np.mean([f[PADS.index(k)][2] > 0 for f in lander["recent_pad_forces"]])))
    sinks = [pads[k]["final_sinkage_m"] for k in PADS if k in touching]
    two_pad_line_distance = None
    if len(touching) == 2:                 # resting on two pads: horizontal distance of the COM from the line through them
        a, b = (np.asarray(final_points[k][0][:2]) for k in touching)
        edge = (b - a) / np.linalg.norm(b - a)
        offset = lander["center_of_mass_now"][:2] - a
        two_pad_line_distance = float(abs(offset[0] * edge[1] - offset[1] * edge[0]))
    results.append(dict(
        idx=site["site_index"], x=site["x"], y=site["y"], yaw_deg=site["yaw_deg"], slope_deg=slope, touchdown_speed_target=arguments.touchdown_speed,
        tilt_deg=tilt, peak_tilt_deg=lander["peak_tilt"], rest_s=lander["rest_time"], touching=touching,
        min_support_margin_m=None if not math.isfinite(lander["min_support_margin"]) else lander["min_support_margin"],
        body_contact=lander["path"] in body_contact_paths, body_contact_parts=sorted(body_contact_parts.get(lander["path"], [])), differential_sinkage_m=(max(sinks) - min(sinks)) if len(sinks) >= 2 else 0.0,
        final_center_of_mass=[float(v) for v in lander["center_of_mass_now"]], weight_N=float(MASS * GRAVITY),
        com_margin_m=margin, two_pad_line_distance_m=two_pad_line_distance, outcome=outcome(tilt, lander["rest_time"], touching, margin, slope),
        final=[float(v) for v in position], pads=pads,
        series_columns=["t_s", "tilt_deg", *[f"sinkage_{k}_m" for k in PADS], *[f"share_{k}" for k in PADS]], series=lander["series"],
        sinkage_definition="depth of the pad-base centre below the undeformed surface, along the terrain normal",
        **(dict(pose_series=lander["pose_series"]) if arguments.record_poses else {})))
    print(f"site {site['site_index']}: tilt {tilt:.2f} deg (peak {lander['peak_tilt']:.2f}), rest {lander['rest_time']}, "
          f"sinkage cm " + ", ".join(f"{k} {100 * pads[k]['final_sinkage_m']:.1f} (max {100 * pads[k]['max_sinkage_m']:.1f})" for k in PADS)
          + f", body contact {sorted(body_contact_parts.get(lander['path'], []))[:6]}", flush=True)

arguments.out.parent.mkdir(parents=True, exist_ok=True)
arguments.out.write_text(json.dumps(dict(
    args=dict(touchdown_speed=arguments.touchdown_speed, clearance=arguments.clearance, pad_radius_m=arguments.pad_radius,
              physics_hz=arguments.physics_hz, soil_mode=arguments.soil_mode, columns_directory=str(columns_directory), sites=arguments.sites,
              mass=float(MASS), soil_carried_parts=len(soil_carried_parts), sinkage_parameters=parameters.to_dict(),
              columns_metadata={str(k): v for k, v in grid_metadata.items()},
              note="DEFORMABLE soil: pressure-sinkage + Janosi shear per footpad (lunar_regolith.contact.sinkage) on SYNTHETIC soil fields; "
                   "terrain surface unchanged (sinkage = depth below it)", provenance=study_paths.git_provenance()),
    results=results)))
print("saved", arguments.out, flush=True)
simulation_application.close()
