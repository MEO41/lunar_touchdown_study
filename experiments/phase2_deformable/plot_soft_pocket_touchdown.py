#!/usr/bin/env python3
"""Footpad depth and lander tilt over time: soft-pocket run vs the rigid run of the same site (plain Python; Phase 2 preview).

    experiments/phase2_deformable/run_phase2.sh plot

Reads results/phase2/soft_pocket_demo/touchdown_{soft_pocket,rigid}_v1.json (written with --record-poses) and writes
docs/images/soft_pocket_touchdown_timeseries.png.  The pocket comes from SYNTHETIC soil fields and its stiffness is ILLUSTRATIVE.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

DEMO_DIRECTORY = study_paths.RESULTS_ROOT / "phase2" / "soft_pocket_demo"
PADS = ("+X", "+Y", "-X", "-Y")
INK, MUTED, SURFACE, GRID = "#1a1a19", "#5f5e5a", "#fcfcfb", "#d9d8d4"
POCKET_COLOR, FIRM_COLOR, RIGID_COLOR = "#2a78d6", "#8a8984", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def tilt_series(run):
    """(time, tilt in degrees) from the recorded poses [t, x, y, z, qw, qx, qy, qz]."""
    poses = np.array(run["pose_series"])
    up_z = 1.0 - 2.0 * (poses[:, 5] ** 2 + poses[:, 6] ** 2)
    return poses[:, 0], np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))


def main():
    """Draw pad depths (both runs) and tilt (both runs) against time."""
    site = json.loads((DEMO_DIRECTORY / "soft_pocket_site.json").read_text())
    pocket_run = json.loads((DEMO_DIRECTORY / "touchdown_soft_pocket_v1.json").read_text())["results"][0]
    rigid_run = json.loads((DEMO_DIRECTORY / "touchdown_rigid_v1.json").read_text())["results"][0]
    pocket_pad = site["site"]["pocket_pad"]
    stop_depth = site["compliance"]["stop_depth_m"]
    figure, (depth_axis, tilt_axis) = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    pocket_depths = np.array(pocket_run["pad_depth_series"])
    rigid_depths = np.array(rigid_run["pad_depth_series"])
    for column, pad in enumerate(PADS, start=1):
        is_pocket = pad == pocket_pad
        depth_axis.plot(pocket_depths[:, 0], 100 * pocket_depths[:, column], lw=2.2 if is_pocket else 1.2,
                        color=POCKET_COLOR if is_pocket else FIRM_COLOR, ls="-" if is_pocket else "--")
    depth_axis.plot(rigid_depths[:, 0], 100 * rigid_depths[:, 1 + PADS.index(pocket_pad)], lw=1.6, color=RIGID_COLOR)
    final_depth = 100 * pocket_depths[-1, 1 + PADS.index(pocket_pad)]
    depth_axis.text(pocket_depths[-1, 0], final_depth + 2, f"pad {pocket_pad} over the pocket: {final_depth:.0f} cm",
                    ha="right", va="bottom", color=INK, fontsize=9)
    depth_axis.text(pocket_depths[-1, 0], 3, f"other three pads (firm): < 2 cm;  pad {pocket_pad} on rigid ground: 0 cm",
                    ha="right", va="bottom", color=MUTED, fontsize=8.5)
    depth_axis.axhline(100 * stop_depth, color=MUTED, lw=0.8, ls=":")
    depth_axis.text(0.1, 100 * stop_depth + 1, f"rigid stop {100 * stop_depth:.0f} cm (firm material under the pocket)", color=MUTED, fontsize=8)
    depth_axis.set_ylim(-2, 100 * stop_depth + 10)
    depth_axis.set_xlabel("time after release (s)")
    depth_axis.set_ylabel("footpad depth below the terrain surface (cm)")
    depth_axis.set_title("Footpad sinkage", fontsize=10, color=INK)
    for run, color, label in ((pocket_run, POCKET_COLOR, "soft pocket under one pad"), (rigid_run, RIGID_COLOR, "rigid ground (Phase 1)")):
        time_s, tilt = tilt_series(run)
        tilt_axis.plot(time_s, tilt, lw=2, color=color, label=label)
    tilt_axis.axhline(pocket_run["slope_deg"], color=MUTED, lw=0.8, ls=":")
    tilt_axis.text(0.1, pocket_run["slope_deg"] + 0.15, f"site slope {pocket_run['slope_deg']:.1f} deg", color=MUTED, fontsize=8)
    tilt_axis.set_xlabel("time after release (s)")
    tilt_axis.set_ylabel("lander tilt (deg)")
    tilt_axis.set_title("Lander tilt", fontsize=10, color=INK)
    tilt_axis.legend(frameon=False, fontsize=9, loc="center right")
    for axis in (depth_axis, tilt_axis):
        axis.grid(True, color=GRID, lw=0.5)
    figure.suptitle(f"Apollo LM, site {site['site']['site_index']}, 1 m/s touchdown: one pad over a soft pocket vs rigid ground\nPocket from "
                    "SYNTHETIC soil fields; compliance ILLUSTRATIVE (not yet from soil mechanics)", fontsize=10.5)
    path = study_paths.FIGURES_DIRECTORY / "soft_pocket_touchdown_timeseries.png"
    figure.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
