#!/usr/bin/env python3
"""Render the soft-pocket touchdown: one footpad sinking into a soft pocket while the other three stay on firm ground (Isaac Sim, RTX).

    experiments/phase2_deformable/run_phase2.sh render        (or: <lander repo>/isaac_run.sh render_soft_pocket_touchdown.py)

No physics runs here: the lander is placed at poses recorded by heterogeneous_touchdown_test.py --record-poses (soft-pocket run and the
rigid run of the same site), and each pose is path-traced.  The terrain is the scene's Near2m tier (LROC NAC DTM, unchanged); a sunken
pad simply disappears below that surface, which is what the compliant pocket faces allow.  The pocket's outline is drawn on the surface
as a thin blue line on the surface (the pocket itself is underground).

Writes results/phase2/soft_pocket_demo/frames/*.png and docs/images/soft_pocket_touchdown_sequence.jpg (contact sheet).
The pocket comes from SYNTHETIC soil fields and its stiffness is ILLUSTRATIVE (see select_soft_pocket_site.py); the captions say so.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

DEMO_DIRECTORY = study_paths.RESULTS_ROOT / "phase2" / "soft_pocket_demo"
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=study_paths.LANDER_SCENE)
parser.add_argument("--lander", type=Path, default=study_paths.LANDER_USD)
parser.add_argument("--site-file", type=Path, default=DEMO_DIRECTORY / "soft_pocket_site.json")
parser.add_argument("--pocket-result", type=Path, default=DEMO_DIRECTORY / "touchdown_soft_pocket_v1.json")
parser.add_argument("--rigid-result", type=Path, default=DEMO_DIRECTORY / "touchdown_rigid_v1.json")
parser.add_argument("--frames", type=Path, default=DEMO_DIRECTORY / "frames")
parser.add_argument("--sheet", type=Path, default=study_paths.FIGURES_DIRECTORY / "soft_pocket_touchdown_sequence.jpg")
parser.add_argument("--resolution", type=int, nargs=2, default=(1600, 900))
parser.add_argument("--subframes", type=int, default=64)
parser.add_argument("--fill", type=float, default=2500.0, help="dim camera-side fill light (0 = off; the sun is 6000)")
arguments = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_application = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": arguments.resolution[0],
                                        "height": arguments.resolution[1]})

import omni.replicator.core as replicator  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

TERRAIN_MESH_PATH = "/World/Terrain/Near2m"
PADS = ("+X", "+Y", "-X", "-Y")


def update(count=1):
    """Advance the application `count` frames."""
    for _ in range(count):
        simulation_application.update()


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
stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
stage.GetPrimAtPath("/World/ApolloLM").SetActive(False)

site_record = json.loads(arguments.site_file.read_text())
pocket_run = json.loads(arguments.pocket_result.read_text())["results"][0]
rigid_run = json.loads(arguments.rigid_result.read_text())["results"][0]
pocket_pad = site_record["site"]["pocket_pad"]
opposite_pad = {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y"}[pocket_pad]

# ---------------------------------------------------------------------------------------------- terrain heights (Near2m grid)
terrain_prim = stage.GetPrimAtPath(TERRAIN_MESH_PATH)
grid = terrain_prim.GetCustomDataByKey("grid")
terrain_points = np.array(UsdGeom.Mesh(terrain_prim).GetPointsAttr().Get(), np.float64)
heights = terrain_points.reshape(grid["ny"], grid["nx"], 3)[:, :, 2]


def terrain_height(x, y):
    """Bilinear height of the Near2m grid at (x, y) (camera placement only)."""
    fx, fy = (x - grid["x0"]) / grid["dx"], (y - grid["y0"]) / grid["dy"]
    i, j = int(np.clip(fx, 0, grid["nx"] - 2)), int(np.clip(fy, 0, grid["ny"] - 2))
    u, v = fx - i, fy - j
    return float((1 - u) * (1 - v) * heights[j, i] + u * (1 - v) * heights[j, i + 1] + (1 - u) * v * heights[j + 1, i] + u * v * heights[j + 1, i + 1])


# ---------------------------------------------------------------------------------------------- pocket outline (visual only)
triangles = np.array(UsdGeom.Mesh(terrain_prim).GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
pocket_triangles = triangles[np.array(site_record["pocket"]["face_indices"], np.int64)]
edges = np.sort(np.concatenate([pocket_triangles[:, [0, 1]], pocket_triangles[:, [1, 2]], pocket_triangles[:, [2, 0]]]), axis=1)
unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
boundary_edges = unique_edges[edge_counts == 1]                      # edges of exactly one pocket triangle = the outline
outline = UsdGeom.BasisCurves.Define(stage, "/World/Visual/SoftPocketOutline")
outline.CreateTypeAttr(UsdGeom.Tokens.linear)
outline.CreateCurveVertexCountsAttr([2] * len(boundary_edges))
outline.CreatePointsAttr([Gf.Vec3f(*map(float, terrain_points[v] + [0, 0, 0.04])) for edge in boundary_edges for v in edge])
outline.CreateWidthsAttr([0.07] * (2 * len(boundary_edges)))
outline.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
outline_material = UsdShade.Material.Define(stage, "/World/Visual/SoftPocketOutlineMaterial")
outline_shader = UsdShade.Shader.Define(stage, "/World/Visual/SoftPocketOutlineMaterial/Shader")
outline_shader.CreateIdAttr("UsdPreviewSurface")
outline_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 1.0))
outline_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.25, 0.8))
outline_material.CreateSurfaceOutput().ConnectToSource(outline_shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(outline.GetPrim()).Bind(outline_material)

# ---------------------------------------------------------------------------------------------- the lander (no physics)
lander = UsdGeom.Xform.Define(stage, "/World/Visual/Lander")
lander.GetPrim().GetReferences().AddReference(str(arguments.lander))
lander_translate = lander.AddTranslateOp()
lander_orient = lander.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
update(10)
for prim in Usd.PrimRange(lander.GetPrim()):          # visual replay only: no rigid body, no colliders
    if prim.HasAPI("PhysicsRigidBodyAPI"):
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    if prim.HasAPI("PhysicsCollisionAPI"):
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)


def place_lander(pose_row):
    """Put the lander at a recorded pose [t, x, y, z, qw, qx, qy, qz]."""
    lander_translate.Set(Gf.Vec3d(*pose_row[1:4]))
    lander_orient.Set(Gf.Quatd(pose_row[4], pose_row[5], pose_row[6], pose_row[7]))


# ---------------------------------------------------------------------------------------------- cameras and light
pocket_pad_xy = np.array(pocket_run["pads"][pocket_pad]["final_contact_east_north"])
opposite_pad_xy = np.array(rigid_run["pads"][opposite_pad]["final_contact_east_north"])
axis = (pocket_pad_xy - opposite_pad_xy) / np.linalg.norm(pocket_pad_xy - opposite_pad_xy)
side = np.array([-axis[1], axis[0]])
middle = 0.5 * (pocket_pad_xy + opposite_pad_xy)
ground_middle = terrain_height(*middle)


def make_camera(name, eye_xy, eye_height, target, focal_length_mm=24.0):
    """A camera at eye_xy, eye_height above the terrain there, looking at `target` (world xyz)."""
    eye = np.array([eye_xy[0], eye_xy[1], terrain_height(*eye_xy) + eye_height])
    camera = UsdGeom.Camera.Define(stage, f"/World/Visual/Cameras/{name}")
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * arguments.resolution[1] / arguments.resolution[0])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1e5))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(view)
    return str(camera.GetPath()), eye


# side view across the sinking pad and its opposite pad (shows the tilt), from whichever side has the clearer line of sight
best_side = max((1.0, -1.0), key=lambda s: min(terrain_height(*(middle + s * side * d)) for d in (6, 10, 14)) * -1)
wide_camera, wide_eye = make_camera("Wide", middle + best_side * 25.0 * side + 2.0 * axis, 1.2, [*middle, ground_middle + 2.2], 40.0)
close_target = [*pocket_pad_xy, terrain_height(*pocket_pad_xy) + 0.3]
close_camera, close_eye = make_camera("Close", pocket_pad_xy + 4.2 * axis + best_side * 2.6 * side, 1.1, close_target)
for name, eye, target in (("Wide", wide_eye, [*middle, ground_middle]), ("Close", close_eye, close_target)):
    if arguments.fill > 0:
        direction = np.array(target) - eye
        direction[2] = 0.0
        direction = 0.35 * direction / max(np.linalg.norm(direction), 1e-6) + np.array([0.0, 0.0, -1.0])
        light = UsdLux.DistantLight.Define(stage, f"/World/Visual/Fill{name}")
        light.CreateIntensityAttr(float(arguments.fill))
        light.CreateAngleAttr(8.0)
        UsdGeom.Xformable(light.GetPrim()).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*(direction / np.linalg.norm(direction)))).GetQuat())
update(30)


def fill_light_on(name):
    """Only the fill light of the camera in use is on."""
    for other in ("Wide", "Close"):
        prim = stage.GetPrimAtPath(f"/World/Visual/Fill{other}")
        if prim:
            UsdGeom.Imageable(prim).MakeVisible() if other == name else UsdGeom.Imageable(prim).MakeInvisible()


# ---------------------------------------------------------------------------------------------- which moments to show
def pose_at(run, time_s):
    """Recorded pose row closest to time_s."""
    return min(run["pose_series"], key=lambda row: abs(row[0] - time_s))


def depth_at(run, time_s):
    """Pad depth row closest to time_s."""
    return min(run["pad_depth_series"], key=lambda row: abs(row[0] - time_s))


pad_column = 1 + PADS.index(pocket_pad)
depths = pocket_run["pad_depth_series"]
deepest = max(row[pad_column] for row in depths)
contact_time = min(p["first_contact_s"] for p in pocket_run["pads"].values() if p["first_contact_s"] is not None)
firm_pads_first = sorted(k for k, p in pocket_run["pads"].items() if p["first_contact_s"] is not None and p["first_contact_s"] < contact_time + 0.1)
pocket_contact_time = pocket_run["pads"][pocket_pad]["first_contact_s"]
half_time = next(row[0] for row in depths if row[pad_column] >= 0.5 * deepest)
rest_time = pocket_run["rest_s"] or depths[-1][0]
rigid_rest_time = rigid_run["rest_s"] or rigid_run["pose_series"][-1][0]
moments = [("1  release, 0.25 m above the ground", pocket_run, 0.0, wide_camera, "Wide"),
           (f"2  first contact: pads {' '.join(firm_pads_first)} (firm ground)", pocket_run, contact_time, wide_camera, "Wide"),
           (f"3  pad {pocket_pad} reaches the pocket", pocket_run, pocket_contact_time, wide_camera, "Wide"),
           (f"4  pad {pocket_pad} sinking", pocket_run, half_time, wide_camera, "Wide"),
           ("5  at rest on the soft pocket", pocket_run, rest_time, wide_camera, "Wide"),
           (f"6  close-up: pad {pocket_pad} at rest", pocket_run, rest_time, close_camera, "Close"),
           ("7  same site, rigid ground (Phase 1)", rigid_run, rigid_rest_time, wide_camera, "Wide"),
           (f"8  close-up: pad {pocket_pad}, rigid ground", rigid_run, rigid_rest_time, close_camera, "Close")]

try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    footer_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except OSError:
    title_font = text_font = footer_font = ImageFont.load_default()
arguments.frames.mkdir(parents=True, exist_ok=True)
width, height = arguments.resolution
frame_paths = []
render_product = annotator = None
current_camera = None
for caption, run, time_s, camera, light_name in moments:
    place_lander(pose_at(run, time_s))
    fill_light_on(light_name)
    update(5)
    if camera != current_camera:
        if render_product is not None:
            annotator.detach()
            render_product.destroy()
        render_product = replicator.create.render_product(camera, (width, height))
        annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        current_camera = camera
    replicator.orchestrator.step(rt_subframes=arguments.subframes, delta_time=0.0)
    image = Image.fromarray(np.clip(np.asarray(annotator.get_data())[..., :3], 0, 255).astype(np.uint8))
    depth_row = depth_at(run, time_s)
    pose_row = pose_at(run, time_s)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (pose_row[5] ** 2 + pose_row[6] ** 2), -1, 1)))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, 0, width, 100], fill=(0, 0, 0, 170))
    draw.text((20, 10), caption, fill=(255, 255, 255, 255), font=title_font)
    draw.text((20, 58), f"t = {time_s:4.1f} s    pad {pocket_pad} depth {100 * depth_row[pad_column]:4.0f} cm    lander tilt {tilt:4.1f} deg    "
                        f"site slope {run['slope_deg']:.1f} deg", fill=(255, 255, 255, 255), font=text_font)
    path = arguments.frames / f"frame_{len(frame_paths) + 1}.png"
    image.save(path)
    frame_paths.append(path)
    print(f"rendered {path.name}: {caption} (t = {time_s:.2f} s)", flush=True)
annotator.detach()
render_product.destroy()

# ---------------------------------------------------------------------------------------------- contact sheet
tiles = [Image.open(p).resize((width // 2, height // 2)) for p in frame_paths]
columns, rows = 4, 2
footer = 70
sheet = Image.new("RGB", (columns * width // 2, rows * height // 2 + footer), (20, 20, 20))
for index, tile in enumerate(tiles):
    sheet.paste(tile, ((index % columns) * width // 2, (index // columns) * height // 2))
draw = ImageDraw.Draw(sheet)
compliance = site_record["compliance"]
draw.text((16, rows * height // 2 + 8), f"Apollo LM, site {site_record['site']['site_index']}, 1 m/s touchdown.  Blue line = outline of a soft pocket "
          f"{site_record['site']['pocket_top_m']:.2f} m below the surface (SYNTHETIC soil field).  Pocket faces compliant "
          f"({compliance['stiffness_per_contact_N_per_m']:.0f} N/m, {compliance['damping_per_contact_N_s_per_m']:.0f} N s/m per contact point; "
          f"rigid stop {compliance['stop_depth_m']} m down):", fill=(235, 235, 235), font=footer_font)
draw.text((16, rows * height // 2 + 36), "ILLUSTRATIVE values, not yet derived from soil mechanics (Phase 2).  Terrain: LROC NAC DTM, unchanged; "
          "all other ground rigid.", fill=(235, 235, 235), font=footer_font)
arguments.sheet.parent.mkdir(parents=True, exist_ok=True)
sheet.save(arguments.sheet, quality=88)
print("saved", arguments.sheet, flush=True)
simulation_application.close()
