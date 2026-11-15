#!/usr/bin/env python3
"""Gate test: does PhysX (Isaac Sim) honour per-face soil friction zones, and what friction does a body really feel? (Isaac Sim Python)

    experiments/phase1_static_friction/run_phase1.sh gate        (or: <lander repo>/isaac_run.sh <this file>)

Before the heterogeneous touchdown experiment trusts the soil friction overlay from lunar_regolith, this checks on simple geometry that:
  1. per-face physics materials authored as GeomSubsets (the exact code path of lunar_regolith's write_friction_overlay) are used by PhysX;
  2. the zone materials' frictionCombineMode = multiply, together with a footpad material of friction 1.0, gives exactly the zone coefficients;
  3. how much friction a body WITHOUT a physics material really gets (PhysX's default material is combined with the terrain material), which is
     the situation of the existing drop-test baseline (the lander has no physics material; the regolith material is 0.8 / 0.7, combine average).

Two tilted triangle-mesh plates (35 deg and 50 deg), each with three strips:
    low zone  (mu 0.46 / 0.406, combine multiply)    high zone (mu 1.10 / 0.966, combine multiply)    uniform regolith (0.8 / 0.7, combine average)
On every strip two footpad-sized slabs (0.9 x 0.9 x 0.2 m, 300 kg): one WITH a footpad material (1.0 / 1.0), one WITHOUT any material.
  35 deg plate (tan = 0.70): static test - a slab must stick when its effective static friction > 0.70 and slide otherwise.
  50 deg plate (tan = 1.19): every slab slides; effective dynamic friction = tan(theta) - a / (g cos(theta)) from the measured acceleration.
Writes results/phase1/friction_gate/friction_zone_physx_check.json and prints a PASS / FAIL table.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

sys.path.insert(0, str(study_paths.REGOLITH_REPOSITORY / "src"))

from isaacsim import SimulationApp  # noqa: E402

simulation_application = SimulationApp({"headless": True})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from lunar_regolith.contact.friction_zones import build_friction_zones  # noqa: E402
from lunar_regolith.contact.usd_terrain import write_friction_overlay  # noqa: E402

LUNAR_GRAVITY_M_PER_S2 = 1.62
LOW_ZONE_FRICTION = (0.46, 0.406)
HIGH_ZONE_FRICTION = (1.10, 0.966)
UNIFORM_REGOLITH_FRICTION = (0.8, 0.7)                     # the lander scene's /Terrain/PhysicsMaterials/Regolith
FOOTPAD_FRICTION = (1.0, 1.0)
STRIP_NAMES = ("low_zone", "high_zone", "uniform_regolith")
STRIP_CENTER_Y_M = {"low_zone": -6.0, "high_zone": 0.0, "uniform_regolith": 6.0}
STRIP_HALF_WIDTH_M = 2.5
PLATE_LENGTH_M = 16.0
PLATE_ANGLES_DEG = (35.0, 50.0)
PLATE_OFFSET_X_M = {35.0: 0.0, 50.0: 40.0}
SLAB_SIZE_M = (0.9, 0.9, 0.2)
SLAB_MASS_KG = 300.0
PHYSICS_STEPS_PER_SECOND = 60


def plate_triangle_mesh(stage, path, angle_deg, offset_x_m):
    """Define one tilted plate (three strips side by side, falling towards +x) as a triangle mesh with an exact triangle-mesh collider.
    Returns the per-face strip name list, in face order."""
    slope = math.tan(math.radians(angle_deg))
    cells_along, cells_across = 16, 5
    points, faces, face_strips = [], [], []
    for strip in STRIP_NAMES:
        y_center = STRIP_CENTER_Y_M[strip]
        base_index = len(points)
        xs = np.linspace(0.0, PLATE_LENGTH_M, cells_along + 1)
        ys = np.linspace(y_center - STRIP_HALF_WIDTH_M, y_center + STRIP_HALF_WIDTH_M, cells_across + 1)
        for y in ys:
            for x in xs:
                points.append((offset_x_m + x, y, -slope * x))
        row_length = cells_along + 1
        for j in range(cells_across):
            for i in range(cells_along):
                a = base_index + j * row_length + i
                b, c, d = a + 1, a + row_length, a + row_length + 1
                faces += [(a, b, d), (a, d, c)]                     # counter-clockwise seen from above
                face_strips += [strip, strip]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(v) for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    return face_strips


def physics_material(stage, path, static_friction, dynamic_friction, combine_mode=None):
    """Define a UsdPhysics material (restitution 0); combine_mode sets PhysX's frictionCombineMode when given."""
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(0.0)
    if combine_mode:
        material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine_mode)
    return material


def slab_rigid_body(stage, path, center, angle_deg, footpad_material):
    """A footpad-sized box rigid body lying flat on the plate surface (rotated by the plate angle about +y)."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    half_angle = math.radians(angle_deg) / 2.0                     # plate falls towards +x: rotate about +y by +angle
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(half_angle), 0.0, math.sin(half_angle), 0.0))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(SLAB_MASS_KG)
    # never sleep: a slab starting from rest on a slope is slow at first and PhysX would freeze it (as in settle_lander.py)
    xform.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    xform.GetPrim().CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    cube = UsdGeom.Cube.Define(stage, path + "/Collider")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*SLAB_SIZE_M))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if footpad_material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(footpad_material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return xform.GetPrim()


def build_stages(work_directory):
    """Write the base stage (plates, uniform material, slabs) and the friction-zone overlay; return the root stage path."""
    base_path = work_directory / "friction_check_base.usda"
    root_path = work_directory / "friction_check_root.usda"
    base = Usd.Stage.CreateNew(str(base_path))
    UsdGeom.SetStageUpAxis(base, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(base, 1.0)
    UsdGeom.Xform.Define(base, "/World")
    scene = UsdPhysics.Scene.Define(base, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(LUNAR_GRAVITY_M_PER_S2)
    uniform_material = physics_material(base, "/World/Materials/UniformRegolith", *UNIFORM_REGOLITH_FRICTION)   # combine average (default)
    footpad_material = physics_material(base, "/World/Materials/Footpad", *FOOTPAD_FRICTION)
    zone_ids_by_plate = {}
    slabs = []
    for angle in PLATE_ANGLES_DEG:
        plate_path = f"/World/Plate{int(angle)}"
        face_strips = plate_triangle_mesh(base, plate_path, angle, PLATE_OFFSET_X_M[angle])
        UsdShade.MaterialBindingAPI.Apply(base.GetPrimAtPath(plate_path)).Bind(uniform_material, UsdShade.Tokens.weakerThanDescendants, "physics")
        zone_ids_by_plate[plate_path] = np.array([0 if s == "low_zone" else 1 if s == "high_zone" else -1 for s in face_strips], dtype=np.int32)
        slope = math.tan(math.radians(angle))
        for strip in STRIP_NAMES:
            for with_material in (True, False):
                x_along = 3.0
                y = STRIP_CENTER_Y_M[strip] + (-1.0 if with_material else 1.0)
                normal = np.array([math.sin(math.radians(angle)), 0.0, math.cos(math.radians(angle))])
                surface_point = np.array([PLATE_OFFSET_X_M[angle] + x_along, y, -slope * x_along])
                center = surface_point + normal * (SLAB_SIZE_M[2] / 2 + 0.005)
                name = f"Slab_{int(angle)}deg_{strip}_{'footpad_material' if with_material else 'no_material'}"
                slabs.append(dict(path=f"/World/{name}", angle_deg=angle, strip=strip, footpad_material=with_material,
                                  start=center.tolist()))
                slab_rigid_body(base, f"/World/{name}", center, angle, footpad_material if with_material else None)
    base.GetRootLayer().Save()
    # friction zones through the SAME authoring code as the lander overlay (lunar_regolith.contact.usd_terrain)
    any_plate = next(iter(zone_ids_by_plate))
    zone_ids = zone_ids_by_plate[any_plate]
    mu_static = np.where(zone_ids == 0, LOW_ZONE_FRICTION[0], np.where(zone_ids == 1, HIGH_ZONE_FRICTION[0], np.nan))
    mu_dynamic = np.where(zone_ids == 0, LOW_ZONE_FRICTION[1], np.where(zone_ids == 1, HIGH_ZONE_FRICTION[1], np.nan))
    built_ids, table = build_friction_zones(mu_static, mu_dynamic, np.degrees(np.arctan(mu_static)), n_zones=2)
    overlay_stage_written = []
    for index, (plate_path, ids) in enumerate(zone_ids_by_plate.items()):
        layer_path = work_directory / f"friction_check_overlay_{index}.usda"
        write_friction_overlay(layer_path, plate_path, built_ids, table, {"purpose": "verify_friction_zones_in_physx"},
                               material_root=f"/World/Materials/SoilFrictionZones{index}")
        overlay_stage_written.append(layer_path)
    root = Usd.Stage.CreateNew(str(root_path))
    root.GetRootLayer().subLayerPaths = [str(p) for p in overlay_stage_written] + [str(base_path)]
    root.SetDefaultPrim(root.GetPrimAtPath("/World"))
    # without a time range the timeline stops after its first frame and physics never advances
    root.SetTimeCodesPerSecond(PHYSICS_STEPS_PER_SECOND)
    root.SetStartTimeCode(0)
    root.SetEndTimeCode(1_000_000)
    root.GetRootLayer().Save()
    return root_path, slabs, table


def main():
    """Run both plates, measure each slab's motion, compare with the expected effective friction and save the report."""
    work_directory = Path(tempfile.mkdtemp(prefix="friction_zone_check_"))
    root_path, slabs, table = build_stages(work_directory)
    context = omni.usd.get_context()
    context.open_stage(str(root_path))
    for _ in range(20):
        simulation_application.update()
    stage = context.get_stage()
    zone_values = {z.zone_id: (z.mu_static, z.mu_dynamic) for z in table.zones}
    print("zones from build_friction_zones:", zone_values, flush=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    samples = {s["path"]: [] for s in slabs}
    total_steps = int(4.0 * PHYSICS_STEPS_PER_SECOND)
    for step in range(total_steps):
        simulation_application.update()
        time_s = (step + 1) / PHYSICS_STEPS_PER_SECOND
        cache = UsdGeom.XformCache()
        for s in slabs:
            translation = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(s["path"])).ExtractTranslation())
            samples[s["path"]].append((time_s, translation))
    timeline.stop()
    if abs(timeline.get_current_time()) < 1e-9 and max(np.linalg.norm(samples[s["path"]][-1][1] - samples[s["path"]][0][1])
                                                     for s in slabs) < 1e-3:
        raise SystemExit("physics did not advance (timeline stuck) - no result")

    def expected_effective(strip, with_material, which):
        """Effective coefficient PhysX should use for this strip / slab material pair (which = 0 static, 1 dynamic)."""
        default_material = 0.5                                     # assumed PhysX default material; measured below
        pad = FOOTPAD_FRICTION[which] if with_material else default_material
        if strip == "low_zone":
            return LOW_ZONE_FRICTION[which] * pad                  # multiply wins over average
        if strip == "high_zone":
            return HIGH_ZONE_FRICTION[which] * pad
        return 0.5 * (UNIFORM_REGOLITH_FRICTION[which] + pad)       # both average

    rows = []
    for s in slabs:
        series = samples[s["path"]]
        times = np.array([t for t, _ in series])
        positions = np.array([p for _, p in series])
        direction = np.array([math.cos(math.radians(s["angle_deg"])), 0.0, -math.sin(math.radians(s["angle_deg"]))])
        along = (positions - positions[0]) @ direction
        window = (times >= 0.5) & (times <= 3.0)
        acceleration = 2.0 * np.polyfit(times[window], along[window], 2)[0]
        angle = math.radians(s["angle_deg"])
        measured_dynamic = math.tan(angle) - acceleration / (LUNAR_GRAVITY_M_PER_S2 * math.cos(angle))
        slid_m = float(along[-1])
        static_expected = expected_effective(s["strip"], s["footpad_material"], 0)
        dynamic_expected = expected_effective(s["strip"], s["footpad_material"], 1)
        row = dict(slab=s["path"].split("/")[-1], angle_deg=s["angle_deg"], strip=s["strip"], footpad_material=s["footpad_material"],
                   slid_m=slid_m, acceleration_m_per_s2=float(acceleration), expected_static=static_expected, expected_dynamic=dynamic_expected)
        if s["angle_deg"] == 35.0:
            should_slide = static_expected < math.tan(angle)
            row.update(expected_to_slide=bool(should_slide), slid=bool(slid_m > 0.02),
                       passed=bool((slid_m > 0.02) == should_slide))
        else:
            row.update(measured_dynamic=float(measured_dynamic), passed=bool(abs(measured_dynamic - dynamic_expected) < 0.03))
        rows.append(row)
    footpad_rows = [r for r in rows if r["footpad_material"] and r["strip"] != "uniform_regolith"]
    zones_honoured = all(r["passed"] for r in footpad_rows)
    report = dict(work_directory=str(work_directory), zones=zone_values, rows=rows, zones_honoured_with_footpad_material=zones_honoured,
                  note="expected values for slabs WITHOUT a material assume PhysX's default material is 0.5 / 0.5; their measured dynamic "
                       "friction shows what the drop-test baseline lander (no physics material) actually experiences")
    report["provenance"] = study_paths.git_provenance()
    output_path = study_paths.phase_results_directory("phase1", "friction_gate") / "friction_zone_physx_check.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"{'slab':52s} {'slid m':>7s} {'expect':>14s} {'measured mu_d':>13s}  result", flush=True)
    for r in rows:
        expectation = (f"slide={r['expected_to_slide']}" if r["angle_deg"] == 35.0 else f"mu_d={r['expected_dynamic']:.3f}")
        measured = f"{r.get('measured_dynamic', float('nan')):.3f}"
        print(f"{r['slab']:52s} {r['slid_m']:7.2f} {expectation:>14s} {measured:>13s}  {'PASS' if r['passed'] else 'FAIL'}", flush=True)
    print(f"per-face friction zones honoured (footpad material, multiply): {'YES' if zones_honoured else 'NO'}", flush=True)
    print("saved", output_path, flush=True)


main()
simulation_application.close()
