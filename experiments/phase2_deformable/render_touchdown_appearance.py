#!/usr/bin/env python3
"""Render a deformable-soil touchdown on the field-driven regolith: the lander at its recorded rest pose, its footpads
in their imprints (Isaac Sim RTX; Phase 2 visualisation).

    experiments/phase2_deformable/run_phase2.sh appearance-render     (or: isaac_run.sh render_touchdown_appearance.py --case <dir>)

<case> is a folder written by build_touchdown_appearance.py. The lander scene's Near2m tier is hidden for rendering (it is
not modified) and the case's render meshes - an exact refinement of Near2m with the footpad imprints, micro-relief,
micro-craters, rocks and clods, and the lunar_regolith appearance material - are referenced in its place. The lander is
a visual copy (no physics) at the last recorded pose. Views: a wide side view across the deepest pad and the pad opposite
it, close-ups of both, and the same three without the lander (the dents at their loaded depth).
Writes <case>/frames/*.png and docs/images/touchdown_appearance_<case>.jpg.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--case", type=Path, required=True, help="build_touchdown_appearance.py output folder")
parser.add_argument("--scene", type=Path, default=study_paths.LANDER_SCENE)
parser.add_argument("--lander", type=Path, default=study_paths.LANDER_USD)
parser.add_argument("--resolution", type=int, nargs=2, default=(1600, 900))
parser.add_argument("--subframes", type=int, default=64)
parser.add_argument("--fill", type=float, default=1500.0, help="dim camera-side fill light (0 = off; the scene's Sun is 6000)")
parser.add_argument("--sheet", type=Path, default=None)
arguments = parser.parse_args()
case = json.loads((arguments.case / "touchdown_appearance.json").read_text())

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing", "width": arguments.resolution[0],
                     "height": arguments.resolution[1]})

import carb  # noqa: E402
import omni.replicator.core as replicator  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

for key in ("/app/viewport/grid/enabled", "/persistent/app/viewport/grid/enabled"):
    carb.settings.get_settings().set(key, False)


def update(count=1):
    for _ in range(count):
        app.update()


context = omni.usd.get_context()
context.open_stage(str(arguments.scene))
update(20)
while context.get_stage_loading_status()[2] > 0:
    update()
stage = context.get_stage()
variants = stage.GetPrimAtPath("/World").GetVariantSets()
variants.GetVariantSet("terrain").SetVariantSelection("nac_2m")
if variants.HasVariantSet("regolith"):
    variants.GetVariantSet("regolith").SetVariantSelection("rigid")
update(20)
stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
stage.GetPrimAtPath("/World/ApolloLM").SetActive(False)

# ---------------------------------------------------------------------------------------------- terrain: render meshes in place of Near2m
near = stage.GetPrimAtPath("/World/Terrain/Near2m")
grid = near.GetCustomDataByKey("grid")
heights = np.array(UsdGeom.Mesh(near).GetPointsAttr().Get(), np.float64).reshape(grid["ny"], grid["nx"], 3)[:, :, 2]
UsdGeom.Imageable(near).MakeInvisible()                 # hidden for rendering only; geometry untouched
appearance = UsdGeom.Xform.Define(stage, "/World/RegolithAppearance")
appearance.GetPrim().GetReferences().AddReference(str((arguments.case / "scene.usd").resolve()))
update(10)
for child in ("Lights", "Cameras"):
    prim = stage.GetPrimAtPath(f"/World/RegolithAppearance/{child}")
    if prim:
        prim.SetActive(False)


def terrain_height(x, y):
    """Bilinear Near2m height (camera placement)."""
    fx, fy = (x - grid["x0"]) / grid["dx"], (y - grid["y0"]) / grid["dy"]
    i, j = int(np.clip(fx, 0, grid["nx"] - 2)), int(np.clip(fy, 0, grid["ny"] - 2))
    u, v = fx - i, fy - j
    return float((1 - u) * (1 - v) * heights[j, i] + u * (1 - v) * heights[j, i + 1] + (1 - u) * v * heights[j + 1, i]
                 + u * v * heights[j + 1, i + 1])


# ---------------------------------------------------------------------------------------------- the lander (visual only)
lander = UsdGeom.Xform.Define(stage, "/World/Visual/Lander")
lander.GetPrim().GetReferences().AddReference(str(arguments.lander))
pose = case["final_pose"]
lander.AddTranslateOp().Set(Gf.Vec3d(*pose[1:4]))
lander.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pose[4], pose[5], pose[6], pose[7]))
update(10)
for prim in Usd.PrimRange(lander.GetPrim()):
    if prim.HasAPI("PhysicsRigidBodyAPI"):
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    if prim.HasAPI("PhysicsCollisionAPI"):
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

# ---------------------------------------------------------------------------------------------- cameras
pads = case["pads"]
deep_pad = max(pads, key=lambda k: pads[k]["final_sinkage_m"])
opposite = {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y"}[deep_pad]
deep_xy, opposite_xy = np.array(pads[deep_pad]["final_point"][:2]), np.array(pads[opposite]["final_point"][:2])
axis = (deep_xy - opposite_xy) / np.linalg.norm(deep_xy - opposite_xy)
side = np.array([-axis[1], axis[0]])
middle = 0.5 * (deep_xy + opposite_xy)
best_side = max((1.0, -1.0), key=lambda s: -min(terrain_height(*(middle + s * side * d)) for d in (6, 10, 14)))


def make_camera(name, eye_xy, eye_height, target, focal_length_mm=24.0):
    """Camera eye_height above the terrain at eye_xy, looking at target."""
    eye = np.array([eye_xy[0], eye_xy[1], terrain_height(*eye_xy) + eye_height])
    camera = UsdGeom.Camera.Define(stage, f"/World/Visual/Cameras/{name}")
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * arguments.resolution[1] / arguments.resolution[0])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1e5))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(view)
    return str(camera.GetPath()), eye


ground_middle = terrain_height(*middle)
views = []
views.append(make_camera("Wide", middle + best_side * 24.0 * side + 2.0 * axis, 1.6, [*middle, ground_middle + 3.0], 24.0))
for pad, xy in ((deep_pad, deep_xy), (opposite, opposite_xy)):
    outward = (xy - middle) / np.linalg.norm(xy - middle)
    target = [*xy, terrain_height(*xy) + 0.1]
    views.append(make_camera(f"Close{pad.replace('+', 'P').replace('-', 'M')}", xy + 3.2 * outward + best_side * 2.0 * side, 1.3,
                             target, 28.0))
if arguments.fill > 0:
    light = UsdLux.DistantLight.Define(stage, "/World/Visual/Fill")
    light.CreateIntensityAttr(float(arguments.fill))
    light.CreateAngleAttr(8.0)
    direction = np.r_[-best_side * side * 0.35, -1.0]
    UsdGeom.Xformable(light.GetPrim()).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*(direction / np.linalg.norm(direction)))).GetQuat())
update(30)


def capture(camera):
    """RGB frame (retries an empty first frame)."""
    product = replicator.create.render_product(camera, tuple(arguments.resolution))
    annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    image = None
    for _ in range(5):
        replicator.orchestrator.step(rt_subframes=arguments.subframes, delta_time=0.0)
        data = np.asarray(annotator.get_data())
        if data.ndim == 3 and data.size:
            image = data[..., :3].copy()
            break
    annotator.detach()
    product.destroy()
    if image is None:
        raise RuntimeError(f"no image from {camera}")
    return image


try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except OSError:
    title_font = text_font = ImageFont.load_default()
frames_dir = arguments.case / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
names = {views[0][0]: "wide side view", views[1][0]: f"pad {deep_pad} (deepest)", views[2][0]: f"pad {opposite} (opposite)"}
frames = []
for lander_shown in (True, False):
    UsdGeom.Imageable(lander.GetPrim()).MakeVisible() if lander_shown else UsdGeom.Imageable(lander.GetPrim()).MakeInvisible()
    update(5)
    for camera, _ in views:
        image = Image.fromarray(np.clip(capture(camera), 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([0, 0, arguments.resolution[0], 92], fill=(0, 0, 0, 165))
        draw.text((18, 10), names[camera] + ("" if lander_shown else " - lander removed (dents at their loaded depth)"),
                  fill=(255, 255, 255), font=title_font)
        draw.text((18, 54), "pad sinkage  " + "   ".join(f"{k} {100 * p['final_sinkage_m']:.1f} cm" for k, p in pads.items())
                  + f"     tilt {case['tilt_deg']:.2f} deg", fill=(235, 235, 235), font=text_font)
        path = frames_dir / f"{Path(camera).name}_{'lander' if lander_shown else 'dents'}.png"
        image.save(path)
        frames.append(path)
        print(f"rendered {path.name}", flush=True)

tiles = [Image.open(p).resize((arguments.resolution[0] // 2, arguments.resolution[1] // 2)) for p in frames]
columns, rows, footer = 3, 2, 64
tw, th = tiles[0].size
sheet = Image.new("RGB", (columns * tw, rows * th + footer), (18, 18, 18))
for index, tile in enumerate(tiles):
    sheet.paste(tile, ((index % columns) * tw, (index // columns) * th))
draw = ImageDraw.Draw(sheet)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17) if title_font != ImageFont.load_default() else text_font
draw.text((14, rows * th + 8), f"Apollo LM, site {case['site_index']}, {case['touchdown_speed']:g} m/s touchdown on {case['soil_mode']} "
          "SYNTHETIC soil (Phase 2, Isaac Sim). Imprint depths = the run's pressure-sinkage results; surface = lander-scene Near2m "
          "(unchanged),", fill=(225, 225, 225), font=small)
draw.text((14, rows * th + 34), "rendered with lunar_regolith's field-driven appearance (assumed mapping). The soft pocket under the "
          f"deepest pad is buried ({deep_pad}); the surface above it looks like the soil around it.", fill=(225, 225, 225), font=small)
sheet_path = arguments.sheet or study_paths.FIGURES_DIRECTORY / f"touchdown_appearance_{arguments.case.name}.jpg"
sheet.save(sheet_path, quality=88)
print(f"contact sheet -> {sheet_path}")
app.close()
