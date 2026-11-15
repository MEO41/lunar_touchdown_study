#!/usr/bin/env python3
"""Touchdown of the Apollo LM on terrain with spatially varying soil friction vs uniform friction (Isaac Sim Python).  Phase 1 baseline.

    experiments/phase1_static_friction/run_phase1.sh touchdown [speeds...]       (both modes; or run this file with <lander repo>/isaac_run.sh)
    <lander repo>/isaac_run.sh <this file> --friction-mode heterogeneous --touchdown-speed 2.0 [--limit 20]

Terrain: the scene's Near2m tier (2 m DTM, exact triangle-mesh collider, rigid).  Friction:
  heterogeneous  the lunar_regolith soil friction overlay (per-face friction zones derived from SYNTHETIC soil fields; 8 zones, area-median 0.80)
                 is sublayered into the session layer, so the scene files are not changed;
  uniform        the scene's regolith material 0.80 / 0.70 everywhere (--uniform-friction changes it, e.g. for a small-perturbation sensitivity control).
In BOTH modes the landers get a footpad contact material (friction 1.0) and the terrain materials use frictionCombineMode = multiply, so the
footpads feel exactly the terrain coefficient (verify_friction_zones_in_physx.py checked this; without it PhysX would average with its default
0.5, which is what the original drop test did: effective 0.65 / 0.60).

Each lander (the scene's mass / COM / inertia, level, the site's yaw) is released with its lowest footpad point --clearance above the ground and
an initial downward velocity chosen so it touches down at --touchdown-speed (optionally plus --horizontal-speed at a fixed random heading per site).
Contacts are recorded every physics step during the impact.  Per landing it stores: touchdown speed, pad contact order and times, per-pad load
share history (split of PhysX's contact impulses - their absolute size carries a solver bias, see force_arrows.py), peak / final tilt, minimum
support margin of the COM during the landing, per-pad slip, the friction zone under each pad, and the final state classified as in
plot_drop_test.py (lander repository).  Output: results/phase1/touchdown/heterogeneous_touchdown_<mode>_v<speed>[_h<h>].json;
annotate_footpad_soil.py adds full soil states.
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

SOIL_CONTACT_DIRECTORY = study_paths.RESULTS_ROOT / "phase1" / "soil_contact"
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=study_paths.LANDER_SCENE)
parser.add_argument("--lander", type=Path, default=study_paths.LANDER_USD)
parser.add_argument("--points", type=Path, default=study_paths.LANDER_NEAR_SITES)
parser.add_argument("--overlay", type=Path, default=SOIL_CONTACT_DIRECTORY / "soil_friction_overlay_World_Terrain_Near2m.usda")
parser.add_argument("--faces", type=Path, default=SOIL_CONTACT_DIRECTORY / "soil_friction_faces_World_Terrain_Near2m.npz")
parser.add_argument("--friction-mode", choices=("heterogeneous", "uniform"), required=True)
parser.add_argument("--uniform-friction", type=float, nargs=2, default=(0.8, 0.7), metavar=("STATIC", "DYNAMIC"),
                    help="terrain friction in uniform mode (default 0.8 0.7)")
parser.add_argument("--touchdown-speed", type=float, default=2.0, help="vertical speed at first contact (m/s)")
parser.add_argument("--horizontal-speed", type=float, default=0.0, help="horizontal speed at release (m/s), fixed random heading per site")
parser.add_argument("--clearance", type=float, default=0.25, help="gap (m) between the lowest footpad point and the ground at release (a few physics steps of free flight, so the pre-contact speed is measurable)")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--limit", type=int, default=None, help="only the first N sites")
parser.add_argument("--batch", type=int, default=25)
parser.add_argument("--max-seconds", type=float, default=30.0, help="simulated seconds per batch")
parser.add_argument("--detailed-seconds", type=float, default=4.0, help="record every physics step for this long after release")
parser.add_argument("--out", type=Path, default=None)
arguments = parser.parse_args()
if arguments.out is None:
    horizontal_suffix = f"_h{arguments.horizontal_speed:g}" if arguments.horizontal_speed > 0 else ""
    arguments.out = study_paths.phase_results_directory("phase1", "touchdown") / f"heterogeneous_touchdown_{arguments.friction_mode}_v{arguments.touchdown_speed:g}{horizontal_suffix}.json"

from isaacsim import SimulationApp  # noqa: E402

simulation_application = SimulationApp({"headless": True})

import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

PADS = ("+X", "+Y", "-X", "-Y")
PHYSICS_STEP_SECONDS = 1.0 / 60.0
PAD_ASSIGNMENT_RADIUS_M = 1.2
LOADED_SHARE = 0.002                    # a pad carrying > 0.2 % of the total vertical contact load is in contact
FOOTPAD_CONTACT_FRICTION = 1.0
UNIFORM_FRICTION = tuple(arguments.uniform_friction)
TERRAIN_MESH_PATH = "/World/Terrain/Near2m"

# ----------------------------------------------------------------------------------------------------------------- stage
context = omni.usd.get_context()
context.open_stage(str(arguments.scene))
for _ in range(20):
    simulation_application.update()
while context.get_stage_loading_status()[2] > 0:
    simulation_application.update()
stage = context.get_stage()
world_variant_sets = stage.GetPrimAtPath("/World").GetVariantSets()
world_variant_sets.GetVariantSet("terrain").SetVariantSelection("nac_2m")
world_variant_sets.GetVariantSet("regolith").SetVariantSelection("rigid")
for _ in range(20):
    simulation_application.update()

session_layer = stage.GetSessionLayer()
stage.SetEditTarget(Usd.EditTarget(session_layer))
if arguments.friction_mode == "heterogeneous":
    if not arguments.overlay.exists():
        raise SystemExit(f"friction overlay not found: {arguments.overlay} (run: run_phase1.sh soil)")
    session_layer.subLayerPaths.append(str(arguments.overlay))
    for _ in range(10):
        simulation_application.update()
    subsets = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(TERRAIN_MESH_PATH)).GetMaterialBindSubsets()
    if len(subsets) < 2:
        raise SystemExit("the friction overlay did not compose onto the terrain mesh")
    print(f"friction overlay: {len(subsets)} zone subsets on {TERRAIN_MESH_PATH}", flush=True)
else:
    regolith_material = stage.GetPrimAtPath("/World/Terrain/PhysicsMaterials/Regolith")
    material_api = UsdPhysics.MaterialAPI(regolith_material)
    material_api.GetStaticFrictionAttr().Set(UNIFORM_FRICTION[0])
    material_api.GetDynamicFrictionAttr().Set(UNIFORM_FRICTION[1])
    regolith_material.AddAppliedSchema("PhysxMaterialAPI")
    regolith_material.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("multiply")
footpad_material = UsdShade.Material.Define(stage, "/World/LanderContactMaterial")
footpad_material_api = UsdPhysics.MaterialAPI.Apply(footpad_material.GetPrim())
footpad_material_api.CreateStaticFrictionAttr(FOOTPAD_CONTACT_FRICTION)
footpad_material_api.CreateDynamicFrictionAttr(FOOTPAD_CONTACT_FRICTION)
footpad_material_api.CreateRestitutionAttr(0.0)

faces = np.load(arguments.faces)
face_tree = cKDTree(np.c_[faces["centroid_east"], faces["centroid_north"]])
face_zone = faces["zone_id"]
face_mu_static = faces["mu_static"]          # continuous soil value of each face (before zoning)
face_mu_dynamic = faces["mu_dynamic"]
# PhysX applies ONE coefficient pair per zone (the zone table of the overlay), not the continuous face value
zone_table = {zone["zone_id"]: zone for zone in json.loads(arguments.overlay.with_suffix(".json").read_text())["zone_table"]["zones"]}

# ------------------------------------------------------------------------------------------------------- terrain heights


class NearTierSurface:
    """Exact surface of the active Near tier mesh (regular grid of the builder, read from its customData)."""

    def __init__(self, prim_path):
        """Read the grid definition and heights of the tier mesh."""
        prim = stage.GetPrimAtPath(prim_path)
        grid = prim.GetCustomDataByKey("grid")
        self.x0, self.y0, self.dx, self.dy, self.nx, self.ny = grid["x0"], grid["y0"], grid["dx"], grid["dy"], grid["nx"], grid["ny"]
        self.heights = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), np.float64).reshape(self.ny, self.nx, 3)[:, :, 2]
        self.half_size = -self.x0

    def height(self, x, y):
        """Height of the triangle surface at (x, y), same triangulation as the builder."""
        i = int(np.clip((x - self.x0) // self.dx, 0, self.nx - 2))
        j = int(np.clip((y - self.y0) // self.dy, 0, self.ny - 2))
        u, v = (x - self.x0 - i * self.dx) / self.dx, (y - self.y0 - j * self.dy) / self.dy
        za, zb, zc, zd = self.heights[j, i], self.heights[j, i + 1], self.heights[j + 1, i], self.heights[j + 1, i + 1]
        return za + u * (zb - za) + v * (zd - zb) if u >= v else za + u * (zd - zc) + v * (zc - za)


near_surface = NearTierSurface(TERRAIN_MESH_PATH)


def plane_slope_deg(x, y, radius=6.0):
    """Slope (deg) of the plane fitted to the terrain within `radius` m of (x, y)."""
    points, heights = [], []
    for r in (0.0, radius / 3, 2 * radius / 3, radius):
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            points.append((x + r * np.cos(a), y + r * np.sin(a)))
            heights.append(near_surface.height(*points[-1]))
    design = np.c_[np.array(points) - [x, y], np.ones(len(points))]
    coefficients, *_ = np.linalg.lstsq(design, np.array(heights), rcond=None)
    return float(np.degrees(np.arctan(np.hypot(coefficients[0], coefficients[1]))))


# ------------------------------------------------------------------------------------------------------- lander geometry
LANDER_PATH = "/World/ApolloLM"
scene_lander = stage.GetPrimAtPath(LANDER_PATH)
mass_api = UsdPhysics.MassAPI(scene_lander)
MASS = mass_api.GetMassAttr().Get()
CENTER_OF_MASS = mass_api.GetCenterOfMassAttr().Get()
INERTIA = mass_api.GetDiagonalInertiaAttr().Get()
center_of_mass_local = np.array(CENTER_OF_MASS, np.float64)
lander_points = np.vstack([np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), np.float64) for p in Usd.PrimRange(scene_lander)
                           if p.IsA(UsdGeom.Mesh) and "/Geometry/" in str(p.GetPath())])
dish_points = lander_points[lander_points[:, 2] < lander_points[:, 2].min() + 0.45]
dish_points = dish_points[np.hypot(dish_points[:, 0], dish_points[:, 1]) > 3.0]
pad_dish_points = {}
for point in dish_points:
    key = ("+X" if point[0] > 0 else "-X") if abs(point[0]) > abs(point[1]) else ("+Y" if point[1] > 0 else "-Y")
    pad_dish_points.setdefault(key, []).append(point)
pad_dish_points = {k: np.array(v)[::6] for k, v in pad_dish_points.items()}
pad_bottom_center_local = {k: np.r_[v[:, :2].mean(0), v[:, 2].min()] for k, v in pad_dish_points.items()}
physics_scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
GRAVITY = float(physics_scenes[0].GetAttribute("physics:gravityMagnitude").Get()) if physics_scenes else 1.62
GRAVITY = GRAVITY if GRAVITY > 0 and math.isfinite(GRAVITY) else 1.62
scene_lander.SetActive(False)

sites = json.loads(arguments.points.read_text())["points"]
for index, site in enumerate(sites):
    site["site_index"] = index
    site["random_heading_deg"] = float(np.random.default_rng([arguments.seed, index]).uniform(0, 360))
if arguments.limit:
    sites = sites[:arguments.limit]
outside = [s for s in sites if max(abs(s["x"]), abs(s["y"])) > near_surface.half_size - 20]
if outside:
    raise SystemExit(f"{len(outside)} sites are outside the Near2m tier (|x|,|y| <= {near_surface.half_size - 20:.0f} m)")
print(f"mode {arguments.friction_mode}; touchdown {arguments.touchdown_speed} m/s vertical, {arguments.horizontal_speed} m/s horizontal; "
      f"lander {MASS:,.1f} kg; {len(sites)} sites; gravity {GRAVITY}", flush=True)


def convex_hull_2d(points):
    """Counter-clockwise convex hull of 2-D points (monotone chain)."""
    points = sorted(map(tuple, points))

    def half_hull(sequence):
        """One half of the monotone-chain hull."""
        hull = []
        for p in sequence:
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
                hull.pop()
            hull.append(p)
        return hull
    lower, upper = half_hull(points), half_hull(points[::-1])
    return lower[:-1] + upper[:-1]


def support_margin_m(contact_points_xy, center_of_mass_xy):
    """Distance from the COM's vertical projection to the nearest support-polygon edge (+ inside); NaN with < 3 supports."""
    if len(contact_points_xy) < 3:
        return float("nan")
    hull = convex_hull_2d(contact_points_xy)
    if len(hull) < 3:
        return float("nan")
    margin = 1e9
    for i in range(len(hull)):
        a, b = np.array(hull[i]), np.array(hull[(i + 1) % len(hull)])
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        margin = min(margin, float(np.dot(np.asarray(center_of_mass_xy) - a, normal)))
    return margin


def zone_at(east, north):
    """Terrain face nearest to (east, north) (overlay map, both modes): zone id, the zone's coefficients (what PhysX applies in
    heterogeneous mode) and the face's continuous soil coefficients (before zoning)."""
    _, index = face_tree.query([east, north])
    zone = int(face_zone[index])
    return (zone, float(zone_table[zone]["mu_static"]), float(zone_table[zone]["mu_dynamic"]),
            float(face_mu_static[index]), float(face_mu_dynamic[index]))


# ------------------------------------------------------------------------------------------------------- contact capture
contact_rows_this_step = []


def on_contact_report(headers, data):
    """Collect (lander index, contact position, impulse) for every lander-terrain contact of this physics step."""
    for header in headers:
        path0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
        path1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
        for lander_path, other_path in ((path0, path1), (path1, path0)):
            if lander_path.startswith("/World/Drops/L") and "/Terrain/" in other_path:
                lander_index = int(lander_path.split("/")[3][1:])
                for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    c = data[k]
                    contact_rows_this_step.append((lander_index, (c.position.x, c.position.y, c.position.z), (c.impulse.x, c.impulse.y, c.impulse.z)))
                break


contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)


class LandingRecorder:
    """Per-lander time history of pad loads, contacts, tilt and support margin during one touchdown."""

    def __init__(self, site, prim, release_height):
        """Start an empty record for the lander `prim` released over `site`."""
        self.site, self.prim, self.release_height = site, prim, release_height
        self.first_contact_time = {}
        self.first_contact_point = {}
        self.last_contact_point = {}
        self.peak_share = {k: 0.0 for k in PADS}
        self.max_imbalance = 0.0
        self.min_support_margin = float("inf")
        self.peak_tilt = 0.0
        self.touchdown_vertical_speed = None
        self.previous_center_height = None
        self.previous_vertical_speed = None
        self.max_share_multi_contact = 0.0
        self.series = []
        self.final_share_frames = []
        self.history = []
        self.rest_time = None
        self.body_contact_seen = False

    def record_step(self, time_s, transform, rows, keep_series, step_seconds):
        """Process one physics step: assign contacts to pads, update load shares, contact times, tilt and support margin.
        The touchdown speed is the vertical speed of the lander origin over the step BEFORE its first contact (the contact step's own
        velocity is already reduced by the impact)."""
        height = float(transform.ExtractTranslation()[2])
        if self.previous_center_height is not None and step_seconds > 0:
            current_vertical_speed = (self.previous_center_height - height) / step_seconds
        else:
            current_vertical_speed = None
        pad_centers = {k: np.array(transform.Transform(Gf.Vec3d(*c))) for k, c in pad_bottom_center_local.items()}
        up = np.array(transform.TransformDir(Gf.Vec3d(0, 0, 1)))
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        self.peak_tilt = max(self.peak_tilt, tilt)
        vertical = {k: 0.0 for k in PADS + ("body",)}
        weighted_point = {k: np.zeros(3) for k in PADS}
        weight = {k: 0.0 for k in PADS}
        sign = 1.0 if sum(i[2] for _, i in rows) >= 0 else -1.0
        for position, impulse in rows:
            position = np.array(position)
            group = min(PADS, key=lambda k: np.hypot(position[0] - pad_centers[k][0], position[1] - pad_centers[k][1]))
            if np.hypot(position[0] - pad_centers[group][0], position[1] - pad_centers[group][1]) > PAD_ASSIGNMENT_RADIUS_M:
                group = "body"
            vertical[group] += sign * impulse[2]
            if group != "body":
                magnitude = float(np.linalg.norm(impulse))
                weighted_point[group] += magnitude * position
                weight[group] += magnitude
        total = sum(max(v, 0.0) for v in vertical.values())
        shares = {k: (max(vertical[k], 0.0) / total if total > 0 else 0.0) for k in PADS + ("body",)}
        if shares["body"] > 0.05:
            self.body_contact_seen = True
        loaded = [k for k in PADS if shares[k] > LOADED_SHARE and weight[k] > 0]
        for k in loaded:
            point = weighted_point[k] / weight[k]
            if k not in self.first_contact_time:
                self.first_contact_time[k] = time_s
                self.first_contact_point[k] = point
                if self.touchdown_vertical_speed is None and self.previous_vertical_speed is not None:
                    self.touchdown_vertical_speed = float(self.previous_vertical_speed)
            self.last_contact_point[k] = point
            self.peak_share[k] = max(self.peak_share[k], shares[k])
        if total > 0 and loaded:
            self.max_imbalance = max(self.max_imbalance, max(shares[k] for k in PADS))
        if len(loaded) >= 3:
            self.max_share_multi_contact = max(self.max_share_multi_contact, max(shares[k] for k in PADS))
            center_of_mass = np.array(transform.Transform(Gf.Vec3d(*center_of_mass_local)))
            margin = support_margin_m([self.last_contact_point[k][:2] for k in loaded], center_of_mass[:2])
            if math.isfinite(margin):
                self.min_support_margin = min(self.min_support_margin, margin)
        self.final_share_frames.append({k: shares[k] for k in PADS + ("body",)})
        self.final_share_frames = self.final_share_frames[-30:]
        if keep_series:
            self.series.append([round(time_s, 4), round(tilt, 3)] + [round(shares[k], 4) for k in PADS + ("body",)])
        self.previous_center_height = height
        self.previous_vertical_speed = current_vertical_speed


# ------------------------------------------------------------------------------------------------------- simulation
timeline = omni.timeline.get_timeline_interface()
results = []
wall_start = time.time()
for batch_start in range(0, len(sites), arguments.batch):
    batch = sites[batch_start:batch_start + arguments.batch]
    timeline.stop()
    if stage.GetPrimAtPath("/World/Drops"):
        stage.RemovePrim("/World/Drops")
    for _ in range(3):
        simulation_application.update()
    UsdGeom.Xform.Define(stage, "/World/Drops")
    recorders = {}
    for offset, site in enumerate(batch):
        lander_index = batch_start + offset
        path = f"/World/Drops/L{lander_index:04d}"
        xform = UsdGeom.Xform.Define(stage, path)
        xform.GetPrim().GetReferences().AddReference(str(arguments.lander))
        yaw = math.radians(site["yaw_deg"])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        all_dish = np.vstack(list(pad_dish_points.values())) @ rotation.T
        ground_minus_point = [near_surface.height(site["x"] + p[0], site["y"] + p[1]) - p[2] for p in all_dish]
        release_height = max(ground_minus_point) + arguments.clearance
        initial_down_speed = math.sqrt(max(arguments.touchdown_speed ** 2 - 2 * GRAVITY * arguments.clearance, 0.0))
        heading = math.radians(site["random_heading_deg"])
        velocity = (arguments.horizontal_speed * math.cos(heading), arguments.horizontal_speed * math.sin(heading), -initial_down_speed)
        xform.AddTranslateOp().Set(Gf.Vec3d(site["x"], site["y"], release_height))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass.CreateMassAttr(float(MASS))
        mass.CreateCenterOfMassAttr(CENTER_OF_MASS)
        mass.CreateDiagonalInertiaAttr(INERTIA)
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        UsdPhysics.RigidBodyAPI(xform.GetPrim()).CreateVelocityAttr(Gf.Vec3f(*map(float, velocity)))
        PhysxSchema.PhysxContactReportAPI.Apply(xform.GetPrim()).CreateThresholdAttr(0.0)
        # never sleep: sleeping bodies send no contact reports, and the resting load distribution is a result (as in settle_lander.py)
        PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim()).CreateSleepThresholdAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(xform.GetPrim()).Bind(footpad_material, UsdShade.Tokens.strongerThanDescendants, "physics")
        site["_release"] = dict(height=float(release_height), velocity=list(map(float, velocity)), heading_deg=site["random_heading_deg"])
        recorders[lander_index] = LandingRecorder(site, xform.GetPrim(), release_height)
    for _ in range(5):
        simulation_application.update()
    contact_rows_this_step.clear()
    timeline.play()
    step = 0
    pose_history = {i: [] for i in recorders}
    while step * PHYSICS_STEP_SECONDS < arguments.max_seconds and any(r.rest_time is None for r in recorders.values()):
        contact_rows_this_step.clear()
        simulation_application.update()
        step += 1
        time_s = step * PHYSICS_STEP_SECONDS
        detailed = time_s <= arguments.detailed_seconds
        if not detailed and step % 10:
            continue
        rows_by_lander = {}
        for lander_index, position, impulse in contact_rows_this_step:
            rows_by_lander.setdefault(lander_index, []).append((position, impulse))
        cache = UsdGeom.XformCache()
        for lander_index, recorder in recorders.items():
            if recorder.rest_time is not None:
                continue
            transform = cache.GetLocalToWorldTransform(recorder.prim)
            recorder.record_step(time_s, transform, rows_by_lander.get(lander_index, []), detailed and step % 2 == 0,
                                 PHYSICS_STEP_SECONDS if detailed else 10 * PHYSICS_STEP_SECONDS)
            if step % 30 == 0:
                translation = np.array(transform.ExtractTranslation())
                up = np.array(transform.TransformDir(Gf.Vec3d(0, 0, 1)))
                pose_history[lander_index].append((translation, up))
                history = pose_history[lander_index][-5:]
                if len(history) == 5 and time_s > 2.0:
                    speed = max(np.linalg.norm(history[j + 1][0] - history[j][0]) for j in range(4)) / 0.5
                    angular = max(np.arccos(np.clip(np.dot(history[j + 1][1], history[j][1]), -1, 1)) for j in range(4)) / 0.5
                    if speed < 0.005 and angular < 0.002:
                        recorder.rest_time = time_s
    timeline_time = timeline.get_current_time()
    cache = UsdGeom.XformCache()
    for lander_index, recorder in recorders.items():
        site = recorder.site
        transform = cache.GetLocalToWorldTransform(recorder.prim)
        translation = np.array(transform.ExtractTranslation())
        up = np.array(transform.TransformDir(Gf.Vec3d(0, 0, 1)))
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        gaps, lowest_point = {}, {}
        for k, points in pad_dish_points.items():
            world = [np.array(transform.Transform(Gf.Vec3d(*p))) for p in points]
            gap = [w[2] - near_surface.height(w[0], w[1]) for w in world]
            i = int(np.argmin(gap))
            gaps[k], lowest_point[k] = float(gap[i]), world[i]
        touching = [k for k in PADS if gaps[k] < 0.03]
        center_of_mass = np.array(transform.Transform(Gf.Vec3d(*center_of_mass_local)))
        final_margin = support_margin_m([lowest_point[k][:2] for k in touching], center_of_mass[:2])
        final_shares = {k: float(np.mean([f[k] for f in recorder.final_share_frames])) if recorder.final_share_frames else 0.0
                        for k in PADS + ("body",)}
        pads = {}
        for k in PADS:
            first = recorder.first_contact_point.get(k)
            last = recorder.last_contact_point.get(k, lowest_point[k])
            zone, mu_static, mu_dynamic, soil_mu_static, soil_mu_dynamic = zone_at(last[0], last[1])
            effective = (mu_static, mu_dynamic) if arguments.friction_mode == "heterogeneous" else UNIFORM_FRICTION
            pads[k] = dict(first_contact_s=recorder.first_contact_time.get(k), first_contact_east_north=None if first is None else [float(first[0]), float(first[1])],
                           final_contact_east_north=[float(last[0]), float(last[1])],
                           slip_m=None if first is None else float(np.hypot(last[0] - first[0], last[1] - first[1])),
                           peak_load_share=recorder.peak_share[k], final_load_share=final_shares[k], final_gap_cm=gaps[k] * 100,
                           friction_zone=zone, zone_mu_static=mu_static, zone_mu_dynamic=mu_dynamic, soil_mu_static=soil_mu_static,
                           soil_mu_dynamic=soil_mu_dynamic, effective_mu_static=float(effective[0]), effective_mu_dynamic=float(effective[1]))
        loaded_mu = [pads[k]["effective_mu_static"] for k in PADS if pads[k]["first_contact_s"] is not None]
        contact_order = sorted(recorder.first_contact_time, key=recorder.first_contact_time.get)
        results.append(dict(
            idx=site["site_index"], x=site["x"], y=site["y"], yaw_deg=site["yaw_deg"], slope_deg=plane_slope_deg(site["x"], site["y"]),
            friction_mode=arguments.friction_mode, touchdown_speed_target=arguments.touchdown_speed, horizontal_speed=arguments.horizontal_speed,
            release=site["_release"], touchdown_speed_measured=recorder.touchdown_vertical_speed, contact_order=contact_order,
            contact_spread_s=(max(recorder.first_contact_time.values()) - min(recorder.first_contact_time.values())) if recorder.first_contact_time else None,
            tilt_deg=tilt, peak_tilt_deg=recorder.peak_tilt, touching=touching, com_margin_m=final_margin,
            min_support_margin_m=None if not math.isfinite(recorder.min_support_margin) else recorder.min_support_margin,
            max_pad_load_share_any=recorder.max_imbalance, max_pad_load_share_three_or_more_pads=recorder.max_share_multi_contact,
            body_contact=recorder.body_contact_seen,
            slide_m=float(np.hypot(translation[0] - site["x"], translation[1] - site["y"])), rest_s=recorder.rest_time,
            final=[float(v) for v in translation], pad_mu_contrast=(max(loaded_mu) - min(loaded_mu)) if len(loaded_mu) >= 2 else 0.0,
            pads=pads, series_columns=["t_s", "tilt_deg", *[f"share_{k}" for k in PADS], "share_body"], series=recorder.series))
    fallen = [r.release_height - cache.GetLocalToWorldTransform(r.prim).ExtractTranslation()[2] for r in recorders.values()]
    if not any(abs(f) > 0.01 for f in fallen) or timeline_time <= 0:
        raise SystemExit("PhysX did not simulate this batch (landers did not move) - check the log for PhysX / CUDA errors")
    done = sum(1 for r in recorders.values() if r.rest_time is not None)
    print(f"batch {batch_start // arguments.batch + 1}/{(len(sites) + arguments.batch - 1) // arguments.batch}: {done}/{len(batch)} at rest "
          f"({time.time() - wall_start:.0f} s wall)", flush=True)
    timeline.stop()

arguments.out.parent.mkdir(parents=True, exist_ok=True)
arguments.out.write_text(json.dumps(dict(
    args=dict(friction_mode=arguments.friction_mode, touchdown_speed=arguments.touchdown_speed, horizontal_speed=arguments.horizontal_speed,
              clearance=arguments.clearance, seed=arguments.seed, points=arguments.points.name, overlay=str(arguments.overlay),
              faces=str(arguments.faces), mass=float(MASS), com=list(map(float, center_of_mass_local)), gravity=GRAVITY,
              footpad_contact_friction=FOOTPAD_CONTACT_FRICTION, uniform_friction=UNIFORM_FRICTION, pad_friction_record="zone_table",
              note="terrain friction from SYNTHETIC soil fields (lunar_regolith, uncalibrated); rigid terrain (Phase 1)",
              provenance=study_paths.git_provenance()),
    results=results)))
print("saved", arguments.out, flush=True)
simulation_application.close()
