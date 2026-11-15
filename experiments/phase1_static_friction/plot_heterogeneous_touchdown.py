#!/usr/bin/env python3
"""Compare touchdowns on uniform vs spatially varying (heterogeneous) soil friction - Phase 1 baseline (plain Python).

    experiments/phase1_static_friction/run_phase1.sh plot          (or: <regolith python> plot_heterogeneous_touchdown.py)

Reads results/phase1/touchdown/heterogeneous_touchdown_<mode>_v<speed>.json (the *_soil.json variant when present) for both friction modes and every speed
found, pairs the landings site by site, and writes:
  results/phase1/heterogeneous_touchdown_summary.json / .csv      per mode and speed: outcomes, tilt, slide, support margin, load split, pad slip
  docs/images/heterogeneous_touchdown_outcomes.png          outcome rates and landing-dynamics medians, uniform vs heterogeneous
  docs/images/heterogeneous_touchdown_paired.png            the same site in both modes: peak tilt and final slide
  docs/images/heterogeneous_touchdown_pad_slip.png          footpad slip against the friction under that pad (heterogeneous runs)
  docs/images/heterogeneous_touchdown_noise_floor.png       site-by-site differences, heterogeneous vs uniform, next to the same mode run twice
Paired statistics per speed: exact McNemar test on toppling, sign test on slide, and the subset of landings whose pads stand on clearly
different friction (contrast > HIGH_CONTRAST).  Controls in results/phase1/repeatability/ (compared with the original case of the same mode and speed):
  heterogeneous_touchdown_<mode>_v<speed>_repeat.json            unchanged re-run: differences would be simulation noise (determinism check)
  heterogeneous_touchdown_<mode>_v<speed>_perturbed_mu<mu>.json  uniform friction nudged (e.g. 0.80 -> 0.82): the sensitivity of the
                                                                 landings to ANY small change, the reference for the heterogeneous effect
Outcome classes follow the lander repository's plot_drop_test.py (toppled = final tilt > 45 deg; upright = at rest on >= 3 pads, COM inside, tilt <= slope + 5 deg).
All terrain friction values come from SYNTHETIC soil fields (lunar_regolith, uncalibrated); the terrain is rigid (Phase 1).
"""
import csv
import json
import math
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
CONTROLS_DIRECTORY = study_paths.phase_results_directory("phase1", "repeatability")
HIGH_CONTRAST = 0.2           # max - min static friction under the four pads
SUMMARY_DIRECTORY = study_paths.phase_results_directory("phase1")
IMAGES = study_paths.FIGURES_DIRECTORY
INK, MUTED, SURFACE, GRID = "#1a1a19", "#5f5e5a", "#fcfcfb", "#d9d8d4"
MODE_COLORS = {"uniform": "#2a78d6", "heterogeneous": "#eb6834"}          # categorical slots 1 and 2 (fixed order)
REPEAT_COLOR = "#8a8984"                                                       # neutral: same mode run twice
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


def correct_pad_friction(data):
    """Pad friction as PhysX applied it, for result files written before the touchdown script recorded the zone table value.

    Those files stored the nearest face's CONTINUOUS soil coefficient as zone_mu_* / effective_mu_*, while PhysX applies one coefficient
    pair per zone.  The zone id was recorded correctly, so the applied value is recovered from the overlay's zone table (no re-run needed;
    the simulation itself was not affected).  Recomputes pad_mu_contrast over the pads that touched down."""
    arguments, results = data["args"], data["results"]
    if arguments.get("pad_friction_record") == "zone_table":
        return results
    table_path = Path(arguments["overlay"]).with_suffix(".json")
    zones = {z["zone_id"]: z for z in json.loads(table_path.read_text())["zone_table"]["zones"]}
    heterogeneous = arguments["friction_mode"] == "heterogeneous"
    for landing in results:
        for pad in landing["pads"].values():
            zone = zones[pad["friction_zone"]]
            pad["soil_mu_static"], pad["soil_mu_dynamic"] = pad["zone_mu_static"], pad["zone_mu_dynamic"]
            pad["zone_mu_static"], pad["zone_mu_dynamic"] = zone["mu_static"], zone["mu_dynamic"]
            if heterogeneous:
                pad["effective_mu_static"], pad["effective_mu_dynamic"] = zone["mu_static"], zone["mu_dynamic"]
        loaded = [p["effective_mu_static"] for p in landing["pads"].values() if p["first_contact_s"] is not None]
        landing["pad_mu_contrast"] = max(loaded) - min(loaded) if len(loaded) >= 2 else 0.0
    return results


def load_results():
    """{(mode, speed): results list} for every result file found (soil-annotated version preferred), pad friction as applied."""
    found = {}
    for path in sorted(RENDERS.glob("heterogeneous_touchdown_*_v*.json")):
        match = re.fullmatch(r"heterogeneous_touchdown_(uniform|heterogeneous)_v([0-9.]+)(_soil)?\.json", path.name)
        if not match:
            continue
        key = (match.group(1), float(match.group(2)))
        if key in found and not match.group(3):
            continue
        found[key] = correct_pad_friction(json.loads(path.read_text()))
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


def load_controls():
    """{(mode, speed, control name): results list} for the control runs ("repeat" or "perturbed_mu<value>")."""
    found = {}
    for path in sorted(CONTROLS_DIRECTORY.glob("heterogeneous_touchdown_*_v*_*.json")):
        match = re.fullmatch(r"heterogeneous_touchdown_(uniform|heterogeneous)_v([0-9.]+)_(repeat|perturbed_mu[0-9.]+)\.json", path.name)
        if match:
            found[(match.group(1), float(match.group(2)), match.group(3))] = correct_pad_friction(json.loads(path.read_text()))
    return found


def control_label(mode, speed, control):
    """Short label of a control comparison for tables and figures."""
    if control == "repeat":
        return f"{mode}\nrun twice\n{speed:g} m/s"
    return f"{mode}\n0.80 vs {control[len('perturbed_mu'):]}\n{speed:g} m/s"


def paired(uniform, heterogeneous):
    """Site-by-site pairs (same site, same speed) of the two friction modes."""
    by_index = {r["idx"]: r for r in uniform}
    return [(by_index[h["idx"]], h) for h in heterogeneous if h["idx"] in by_index]


def exact_two_sided_binomial_p(successes, trials):
    """Two-sided exact binomial test against p = 0.5 (McNemar's exact test on discordant pairs, or a sign test)."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1)) / 2.0 ** trials
    return min(1.0, 2.0 * tail)


def pair_statistics(pairs):
    """Differences between two runs of the same sites (first, second): outcome changes, McNemar on toppling, sign test on slide, tilt differences."""
    first_outcomes, second_outcomes = [outcome(a) for a, _ in pairs], [outcome(b) for _, b in pairs]
    toppled_only_second = sum(a != "toppled" and b == "toppled" for a, b in zip(first_outcomes, second_outcomes))
    toppled_only_first = sum(a == "toppled" and b != "toppled" for a, b in zip(first_outcomes, second_outcomes))
    slide_differences = np.array([b["slide_m"] - a["slide_m"] for a, b in pairs])
    slide_higher, slide_lower = int((slide_differences > 0).sum()), int((slide_differences < 0).sum())
    tilt_differences = np.array([b["peak_tilt_deg"] - a["peak_tilt_deg"] for a, b in pairs])
    return dict(
        pairs=len(pairs), outcome_changes=int(sum(a != b for a, b in zip(first_outcomes, second_outcomes))),
        toppled_only_second=toppled_only_second, toppled_only_first=toppled_only_first,
        toppling_mcnemar_exact_p=exact_two_sided_binomial_p(toppled_only_second, toppled_only_second + toppled_only_first),
        slide_higher_in_second=slide_higher, slide_lower_in_second=slide_lower,
        slide_sign_test_p=exact_two_sided_binomial_p(slide_higher, slide_higher + slide_lower),
        slide_difference_median_m=median(slide_differences),
        peak_tilt_absolute_difference_median_deg=median(np.abs(tilt_differences)),
        peak_tilt_absolute_difference_p90_deg=percentile(np.abs(tilt_differences), 90),
        identical_peak_tilt=int((np.abs(tilt_differences) < 1e-6).sum()))


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
        slip = np.maximum(np.array([p["slip_m"] for r in heterogeneous for p in r["pads"].values() if p["slip_m"] is not None]), 1e-3)
        if mu.size:
            axis.scatter(mu, slip, s=6, color=MODE_COLORS["heterogeneous"], alpha=0.45, linewidth=0)
            centers, medians = [], []
            for zone_value in np.unique(mu):                     # one point per friction zone (PhysX applies discrete zone values)
                inside = mu == zone_value
                if inside.sum() > 5:
                    centers.append(float(zone_value))
                    medians.append(float(np.median(slip[inside])))
            axis.plot(centers, medians, color=INK, lw=2, marker="o", ms=4, label="median per friction zone")
        uniform_slip = median([p["slip_m"] for r in found.get(("uniform", speed), []) for p in r["pads"].values() if p["slip_m"] is not None])
        if np.isfinite(uniform_slip):
            axis.axhline(uniform_slip, color=MODE_COLORS["uniform"], lw=1.5, ls="--", label=f"uniform-run median ({uniform_slip:.2f} m)")
        axis.set_yscale("log")
        axis.set_xlabel("static friction applied under the pad (zone value)")
        axis.set_ylabel("footpad slip from first contact to rest (m, log)")
        axis.set_title(f"touchdown {speed:g} m/s", fontsize=10, color=INK)
        axis.grid(True, color=GRID, lw=0.5)
        axis.legend(fontsize=8, frameon=False, loc="upper right")
    figure.suptitle("Footpad slip vs local soil friction (heterogeneous runs). SYNTHETIC soil friction, rigid terrain", fontsize=11)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def noise_floor_figure(comparisons, path):
    """Site-by-site differences: heterogeneous vs uniform per speed, next to the same mode run twice (simulation noise floor)."""
    labels = [label for label, _, _ in comparisons]
    colors = [REPEAT_COLOR if "twice" in label else MODE_COLORS["uniform"] if is_control else MODE_COLORS["heterogeneous"]
              for label, _, is_control in comparisons]
    panels = (("outcome_changes", "Sites whose outcome class changed (of 300)"),
              ("peak_tilt_absolute_difference_median_deg", "Median |peak tilt difference| (deg)"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, (key, title) in zip(axes[:2], panels):
        values = [statistics[key] for _, statistics, _ in comparisons]
        bars = axis.bar(range(len(values)), values, 0.7, color=colors)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}" if isinstance(value, float) else str(value),
                      ha="center", va="bottom", fontsize=8, color=INK)
        axis.set_xticks(range(len(values)), labels, fontsize=7.5)
        axis.set_title(title, fontsize=10, color=INK)
        axis.grid(True, axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    axis = axes[2]
    for label, statistics, is_control in comparisons:
        if "twice" in label:
            continue                       # identical re-run: all differences are zero, nothing to draw on a log axis
        if not is_control and not label.endswith("3 m/s"):
            continue                       # compare like with like: the controls are at 3 m/s
        differences = np.sort(np.abs(statistics["peak_tilt_differences"]))
        axis.plot(np.maximum(differences, 1e-4), np.linspace(0, 1, differences.size), lw=2,
                  color=MODE_COLORS["uniform"] if is_control else MODE_COLORS["heterogeneous"], label=label.replace("\n", " "))
    axis.set_xscale("log")
    axis.set_xlabel("|peak tilt difference| at the same site (deg, log)")
    axis.set_ylabel("fraction of sites")
    axis.set_title("Site-by-site |peak tilt difference|, 3 m/s", fontsize=10, color=INK)
    axis.grid(True, color=GRID, lw=0.5)
    axis.legend(fontsize=7, frameon=False, loc="lower right")
    figure.suptitle("Is the heterogeneous-friction effect larger than the landings' sensitivity to any small change?\n"
                    "orange: heterogeneous vs uniform 0.80;  blue: uniform 0.80 vs 0.82;  grey: identical re-run.  "
                    "SYNTHETIC soil friction, rigid terrain, same 300 sites", fontsize=11)
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
                outcome_changes=sorted({f"{outcome(a)} -> {outcome(b)}" for a, b in pairs if outcome(a) != outcome(b)}),
                statistics=pair_statistics(pairs),
                high_contrast=dict(threshold=HIGH_CONTRAST, **pair_statistics([(a, b) for a, b in pairs if b["pad_mu_contrast"] > HIGH_CONTRAST])))
    controls = load_controls()
    comparisons = []
    for speed in speeds:
        pairs = paired(found.get(("uniform", speed), []), found.get(("heterogeneous", speed), []))
        if pairs:
            comparisons.append((f"heterogeneous\nvs uniform\n{speed:g} m/s", dict(pair_statistics(pairs), peak_tilt_differences=[
                b["peak_tilt_deg"] - a["peak_tilt_deg"] for a, b in pairs]), False))
    for (mode, speed, control), control_results in sorted(controls.items()):
        pairs = paired(found.get((mode, speed), []), control_results)
        if pairs:
            summary[f"control_{mode}_v{speed:g}_{control}"] = pair_statistics(pairs)
            comparisons.append((control_label(mode, speed, control), dict(pair_statistics(pairs), peak_tilt_differences=[
                b["peak_tilt_deg"] - a["peak_tilt_deg"] for a, b in pairs]), True))
    summary["provenance"] = study_paths.git_provenance()
    (SUMMARY_DIRECTORY / "heterogeneous_touchdown_summary.json").write_text(json.dumps(summary, indent=2))
    del summary["provenance"]
    rows = [dict(case=k, **v) for k, v in summary.items() if not k.startswith(("paired", "control"))]
    with open(SUMMARY_DIRECTORY / "heterogeneous_touchdown_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    IMAGES.mkdir(parents=True, exist_ok=True)
    outcome_figure(summary, speeds, IMAGES / "heterogeneous_touchdown_outcomes.png")
    paired_figure(found, speeds, IMAGES / "heterogeneous_touchdown_paired.png")
    pad_slip_figure(found, speeds, IMAGES / "heterogeneous_touchdown_pad_slip.png")
    if comparisons:
        noise_floor_figure(comparisons, IMAGES / "heterogeneous_touchdown_noise_floor.png")
    print(f"{'case':26s} {'n':>4s} {'upright':>7s} {'tilted':>6s} {'toppled':>7s} {'moving':>6s} {'v_td':>5s} {'tilt95':>6s} {'slide':>6s} "
          f"{'padslip':>7s} {'margin05':>8s} {'maxshare':>8s}")
    for key, s in summary.items():
        if key.startswith(("paired", "control")):
            continue
        print(f"{key:26s} {s['landings']:4d} {s['upright']:7d} {s['tilted']:6d} {s['toppled']:7d} {s['moving']:6d} {s['touchdown_speed_median']:5.2f} "
              f"{s['peak_tilt_p95_deg']:6.1f} {s['slide_median_m']:6.2f} {s['pad_slip_median_m']:7.3f} {s['min_support_margin_p05_m']:8.2f} "
              f"{s['max_pad_share_three_pads_median']:8.2f}")
    for key, s in summary.items():
        if key.startswith("paired"):
            print(f"{key}: same outcome {s['same_outcome']}/{s['pairs']}; median diff (heterogeneous - uniform): peak tilt "
                  f"{s['peak_tilt_difference_median_deg']:+.2f} deg, slide {s['slide_difference_median_m']:+.3f} m, min margin "
                  f"{s['min_margin_difference_median_m']:+.3f} m; changes: {s['outcome_changes']}")
    print(f"{'comparison (first -> second)':34s} {'pairs':>5s} {'changed':>7s} {'topple+':>7s} {'topple-':>7s} {'McNemar p':>9s} "
          f"{'slide up/down':>13s} {'sign p':>7s} {'|dtilt| med':>11s} {'identical':>9s}")
    rows = [(f"uniform -> heterogeneous {k[len('paired_'):]}", s["statistics"]) for k, s in summary.items() if k.startswith("paired")]
    rows += [(f"  same, contrast > {HIGH_CONTRAST:g} {k[len('paired_'):]}", s["high_contrast"]) for k, s in summary.items() if k.startswith("paired")]
    rows += [(f"control {k[len('control_'):]}", s) for k, s in summary.items() if k.startswith("control")]
    for label, s in rows:
        print(f"{label:34s} {s['pairs']:5d} {s['outcome_changes']:7d} {s['toppled_only_second']:7d} {s['toppled_only_first']:7d} "
              f"{s['toppling_mcnemar_exact_p']:9.3f} {s['slide_higher_in_second']:6d}/{s['slide_lower_in_second']:<6d} {s['slide_sign_test_p']:7.3f} "
              f"{s['peak_tilt_absolute_difference_median_deg']:11.3f} {s['identical_peak_tilt']:9d}")


if __name__ == "__main__":
    main()
