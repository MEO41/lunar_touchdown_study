#!/usr/bin/env python3
"""Compare touchdowns on uniform vs spatially varying (heterogeneous) soil friction - Phase 1 baseline (plain Python).

    experiments/phase1_static_friction/run_phase1.sh plot          (or: <regolith python> plot_heterogeneous_touchdown.py)

Reads results/phase1/touchdown/heterogeneous_touchdown_<mode>_v<speed>.json (the *_soil.json variant when present) for both friction modes and every speed
found, pairs the landings site by site, and writes:
  results/phase1/heterogeneous_touchdown_summary.json / .csv      per mode and speed: outcomes, tilt, slide, support margin, load split, pad slip
  docs/images/heterogeneous_touchdown_outcomes.png          outcome rates and landing-dynamics medians, uniform vs heterogeneous
  docs/images/heterogeneous_touchdown_paired.png            the same site in both modes: peak tilt and final slide
  docs/images/heterogeneous_touchdown_pad_slip.png          footpad slip against the friction under that pad (heterogeneous runs)
Outcome classes follow the lander repository's plot_drop_test.py (toppled = final tilt > 45 deg; upright = at rest on >= 3 pads, COM inside, tilt <= slope + 5 deg).
All terrain friction values come from SYNTHETIC soil fields (lunar_regolith, uncalibrated); the terrain is rigid (Phase 1).
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import study_paths  # noqa: E402

RENDERS = study_paths.phase_results_directory("phase1", "touchdown")
SUMMARY_DIRECTORY = study_paths.phase_results_directory("phase1")
IMAGES = study_paths.FIGURES_DIRECTORY
INK, MUTED, SURFACE, GRID = "#1a1a19", "#5f5e5a", "#fcfcfb", "#d9d8d4"
MODE_COLORS = {"uniform": "#2a78d6", "heterogeneous": "#eb6834"}          # categorical slots 1 and 2 (fixed order)
MODE_LABELS = {"uniform": "uniform friction 0.80 / 0.70", "heterogeneous": "heterogeneous soil friction (median 0.80)"}
PADS = ("+X", "+Y", "-X", "-Y")
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def outcome(landing, tolerance_deg=5.0):
    """Outcome class of one landing, as in plot_drop_test.py."""
    if landing["tilt_deg"] > 45:
        return "toppled"
    if landing["rest_s"] is None:
        return "moving"
    margin = landing["com_margin_m"]
    upright = len(landing["touching"]) >= 3 and margin == margin and margin > 0 and landing["tilt_deg"] <= landing["slope_deg"] + tolerance_deg
    return "upright" if upright else "tilted"


def load_results():
    """{(mode, speed): results list} for every result file found (soil-annotated version preferred)."""
    found = {}
    for path in sorted(RENDERS.glob("heterogeneous_touchdown_*_v*.json")):
        match = re.fullmatch(r"heterogeneous_touchdown_(uniform|heterogeneous)_v([0-9.]+)(_soil)?\.json", path.name)
        if not match:
            continue
        key = (match.group(1), float(match.group(2)))
        if key in found and not match.group(3):
            continue
        found[key] = json.loads(path.read_text())["results"]
    return found


def median(values):
    """Median of the finite values (NaN when there are none)."""
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(values)) if values.size else float("nan")


def percentile(values, q):
    """q-th percentile of the finite values (NaN when there are none)."""
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.percentile(values, q)) if values.size else float("nan")


def summarise(results):
    """Outcome counts and landing-dynamics statistics for one mode and speed."""
    classes = [outcome(r) for r in results]
    pad_slips = [p["slip_m"] for r in results for p in r["pads"].values() if p["slip_m"] is not None]
    return dict(
        landings=len(results), **{f"{c}": classes.count(c) for c in ("upright", "tilted", "toppled", "moving")},
        toppled_percent=100.0 * classes.count("toppled") / max(len(results), 1),
        upright_percent=100.0 * classes.count("upright") / max(len(results), 1),
        touchdown_speed_median=median([r["touchdown_speed_measured"] for r in results]),
        peak_tilt_median_deg=median([r["peak_tilt_deg"] for r in results]), peak_tilt_p95_deg=percentile([r["peak_tilt_deg"] for r in results], 95),
        tilt_over_slope_median_deg=median([r["tilt_deg"] - r["slope_deg"] for r in results]),
        slide_median_m=median([r["slide_m"] for r in results]), slide_p95_m=percentile([r["slide_m"] for r in results], 95),
        min_support_margin_median_m=median([r["min_support_margin_m"] for r in results]),
        min_support_margin_p05_m=percentile([r["min_support_margin_m"] for r in results], 5),
        max_pad_share_three_pads_median=median([r["max_pad_load_share_three_or_more_pads"] for r in results]),
        pad_slip_median_m=median(pad_slips), pad_slip_p95_m=percentile(pad_slips, 95),
        contact_spread_median_s=median([r["contact_spread_s"] for r in results]),
        pad_friction_contrast_median=median([r["pad_mu_contrast"] for r in results]))


def paired(uniform, heterogeneous):
    """Site-by-site pairs (same site, same speed) of the two friction modes."""
    by_index = {r["idx"]: r for r in uniform}
    return [(by_index[h["idx"]], h) for h in heterogeneous if h["idx"] in by_index]


def outcome_figure(summary, speeds, path):
    """Outcome rates and dynamics medians per speed, uniform vs heterogeneous (grouped bars, direct labels)."""
    panels = [("toppled_percent", "Toppled (%)"), ("upright_percent", "Upright at rest (%)"), ("peak_tilt_p95_deg", "Peak tilt, 95th pct (deg)"),
              ("slide_median_m", "Median slide (m)"), ("pad_slip_median_m", "Median footpad slip (m)"),
              ("min_support_margin_p05_m", "Support margin, 5th pct (m)")]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    width = 0.38
    x = np.arange(len(speeds))
    for axis, (key, title) in zip(axes.ravel(), panels):
        for offset, mode in ((-width / 2, "uniform"), (width / 2, "heterogeneous")):
            values = [summary.get(f"{mode}_v{s:g}", {}).get(key, np.nan) for s in speeds]
            bars = axis.bar(x + offset, values, width * 0.92, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}",
                              ha="center", va="bottom", fontsize=8, color=INK)
        axis.set_xticks(x, [f"{s:g} m/s" for s in speeds])
        axis.set_title(title, fontsize=10, color=INK)
        axis.grid(True, axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, loc="upper left", frameon=False)
    figure.suptitle("Apollo LM touchdowns on the Near2m tier (300 sites per bar): uniform vs heterogeneous soil friction\n"
                    "rigid terrain (Phase 1); heterogeneous friction derived from SYNTHETIC, uncalibrated soil fields", fontsize=11)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def paired_figure(found, speeds, path):
    """Peak tilt and slide of the same site in both modes (one row per quantity, one column per speed)."""
    figure, axes = plt.subplots(2, len(speeds), figsize=(4.2 * len(speeds), 8), constrained_layout=True, squeeze=False)
    for column, speed in enumerate(speeds):
        pairs = paired(found.get(("uniform", speed), []), found.get(("heterogeneous", speed), []))
        for row, (key, label) in enumerate((("peak_tilt_deg", "peak tilt (deg)"), ("slide_m", "slide (m)"))):
            axis = axes[row, column]
            u = np.array([a[key] for a, _ in pairs])
            h = np.array([b[key] for _, b in pairs])
            if u.size:
                limit = float(max(u.max(), h.max()) * 1.05 + 1e-9)
                axis.plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
                axis.scatter(u, h, s=14, color=MODE_COLORS["heterogeneous"], edgecolor=SURFACE, linewidth=0.5)
                higher, lower = int((h > u * 1.1 + 0.05).sum()), int((h < u * 0.9 - 0.05).sum())
                axis.text(0.03, 0.97, f"heterogeneous higher: {higher}\nlower: {lower}\nn = {len(pairs)}", transform=axis.transAxes,
                          va="top", fontsize=8, color=INK)
            axis.set_xlabel(f"uniform friction: {label}")
            axis.set_ylabel(f"heterogeneous: {label}")
            axis.set_title(f"touchdown {speed:g} m/s", fontsize=10, color=INK)
            axis.grid(True, color=GRID, lw=0.5)
    figure.suptitle("Same site, both friction modes (dashed: equal). SYNTHETIC soil friction, rigid terrain", fontsize=11)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def pad_slip_figure(found, speeds, path):
    """Footpad slip against the friction coefficient under that pad (heterogeneous runs), with the uniform-run median as reference."""
    figure, axes = plt.subplots(1, len(speeds), figsize=(4.4 * len(speeds), 4.2), constrained_layout=True, squeeze=False)
    for column, speed in enumerate(speeds):
        axis = axes[0, column]
        heterogeneous = found.get(("heterogeneous", speed), [])
        mu = np.array([p["effective_mu_static"] for r in heterogeneous for p in r["pads"].values() if p["slip_m"] is not None])
        slip = np.array([p["slip_m"] for r in heterogeneous for p in r["pads"].values() if p["slip_m"] is not None])
        if mu.size:
            axis.scatter(mu, slip, s=6, color=MODE_COLORS["heterogeneous"], alpha=0.45, linewidth=0)
            edges = np.quantile(mu, np.linspace(0, 1, 9))
            centers, medians = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                inside = (mu >= a) & (mu <= b)
                if inside.sum() > 5:
                    centers.append(float(np.median(mu[inside])))
                    medians.append(float(np.median(slip[inside])))
            axis.plot(centers, medians, color=INK, lw=2, marker="o", ms=4, label="median per friction bin")
        uniform_slip = median([p["slip_m"] for r in found.get(("uniform", speed), []) for p in r["pads"].values() if p["slip_m"] is not None])
        if np.isfinite(uniform_slip):
            axis.axhline(uniform_slip, color=MODE_COLORS["uniform"], lw=1.5, ls="--", label=f"uniform-run median ({uniform_slip:.2f} m)")
        axis.set_xlabel("static friction under the pad (zone)")
        axis.set_ylabel("footpad slip from first contact to rest (m)")
        axis.set_title(f"touchdown {speed:g} m/s", fontsize=10, color=INK)
        axis.grid(True, color=GRID, lw=0.5)
        axis.legend(fontsize=8, frameon=False, loc="upper right")
    figure.suptitle("Footpad slip vs local soil friction (heterogeneous runs). SYNTHETIC soil friction, rigid terrain", fontsize=11)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    """Load all results, write the summary files and the three figures, print a compact table."""
    found = load_results()
    if not found:
        raise SystemExit("no results/phase1/touchdown/heterogeneous_touchdown_*_v*.json found - run run_phase1.sh touchdown first")
    speeds = sorted({speed for _, speed in found})
    summary = {f"{mode}_v{speed:g}": summarise(results) for (mode, speed), results in sorted(found.items())}
    for speed in speeds:
        pairs = paired(found.get(("uniform", speed), []), found.get(("heterogeneous", speed), []))
        if pairs:
            same = sum(outcome(a) == outcome(b) for a, b in pairs)
            summary[f"paired_v{speed:g}"] = dict(
                pairs=len(pairs), same_outcome=same, same_outcome_percent=100.0 * same / len(pairs),
                peak_tilt_difference_median_deg=median([b["peak_tilt_deg"] - a["peak_tilt_deg"] for a, b in pairs]),
                slide_difference_median_m=median([b["slide_m"] - a["slide_m"] for a, b in pairs]),
                min_margin_difference_median_m=median([(b["min_support_margin_m"] or np.nan) - (a["min_support_margin_m"] or np.nan) for a, b in pairs]),
                outcome_changes=sorted({f"{outcome(a)} -> {outcome(b)}" for a, b in pairs if outcome(a) != outcome(b)}))
    summary["provenance"] = study_paths.git_provenance()
    (SUMMARY_DIRECTORY / "heterogeneous_touchdown_summary.json").write_text(json.dumps(summary, indent=2))
    del summary["provenance"]
    rows = [dict(case=k, **v) for k, v in summary.items() if not k.startswith("paired")]
    with open(SUMMARY_DIRECTORY / "heterogeneous_touchdown_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    IMAGES.mkdir(parents=True, exist_ok=True)
    outcome_figure(summary, speeds, IMAGES / "heterogeneous_touchdown_outcomes.png")
    paired_figure(found, speeds, IMAGES / "heterogeneous_touchdown_paired.png")
    pad_slip_figure(found, speeds, IMAGES / "heterogeneous_touchdown_pad_slip.png")
    print(f"{'case':26s} {'n':>4s} {'upright':>7s} {'tilted':>6s} {'toppled':>7s} {'moving':>6s} {'v_td':>5s} {'tilt95':>6s} {'slide':>6s} "
          f"{'padslip':>7s} {'margin05':>8s} {'maxshare':>8s}")
    for key, s in summary.items():
        if key.startswith("paired"):
            continue
        print(f"{key:26s} {s['landings']:4d} {s['upright']:7d} {s['tilted']:6d} {s['toppled']:7d} {s['moving']:6d} {s['touchdown_speed_median']:5.2f} "
              f"{s['peak_tilt_p95_deg']:6.1f} {s['slide_median_m']:6.2f} {s['pad_slip_median_m']:7.3f} {s['min_support_margin_p05_m']:8.2f} "
              f"{s['max_pad_share_three_pads_median']:8.2f}")
    for key, s in summary.items():
        if key.startswith("paired"):
            print(f"{key}: same outcome {s['same_outcome']}/{s['pairs']}; median diff (heterogeneous - uniform): peak tilt "
                  f"{s['peak_tilt_difference_median_deg']:+.2f} deg, slide {s['slide_difference_median_m']:+.3f} m, min margin "
                  f"{s['min_margin_difference_median_m']:+.3f} m; changes: {s['outcome_changes']}")


if __name__ == "__main__":
    main()
