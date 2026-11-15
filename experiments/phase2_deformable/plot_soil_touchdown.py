#!/usr/bin/env python3
"""Deformable-soil touchdowns: heterogeneous soil (one pad over a soft pocket) vs the uniform control (plain Python; Phase 2).

    experiments/phase2_deformable/run_phase2.sh soil-plot

Reads results/phase2/soil_touchdown/soil_touchdown_<heterogeneous|uniform>_v<speed>.json and the candidate list (which pad of each site
stands over the pocket), and writes
  results/phase2/soil_touchdown_summary.json / .csv     per site, speed and soil mode: pocket-pad and other-pad sinkage, differential
                                                         sinkage, final and peak tilt, load share of the pocket pad, outcome
  docs/images/soil_touchdown_comparison.png             the same, as paired figures
Soil fields are SYNTHETIC; the pressure-sinkage constants are lunar literature values applied to them (lunar_regolith.contact.sinkage).
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

RESULTS = study_paths.RESULTS_ROOT / "phase2" / "soil_touchdown"
SITE_FILE = study_paths.RESULTS_ROOT / "phase2" / "soft_pocket_demo" / "soft_pocket_site.json"
INK, MUTED, SURFACE, GRID = "#1a1a19", "#5f5e5a", "#fcfcfb", "#d9d8d4"
MODE_COLORS = {"uniform": "#2a78d6", "heterogeneous": "#eb6834"}
PADS = ("+X", "+Y", "-X", "-Y")
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def load():
    """[(mode, speed, landing)] for every result file."""
    rows = []
    for path in sorted(RESULTS.glob("soil_touchdown_*_v*.json")):
        match = re.fullmatch(r"soil_touchdown_(heterogeneous|uniform)_v([0-9.]+)\.json", path.name)
        if match:
            for landing in json.loads(path.read_text())["results"]:
                rows.append((match.group(1), float(match.group(2)), landing))
    return rows


def summarise(rows, pocket_pad_of_site):
    """One flat record per landing."""
    records = []
    for mode, speed, landing in rows:
        pocket_pad = pocket_pad_of_site[landing["idx"]]
        pads = landing["pads"]
        others = [k for k in PADS if k != pocket_pad]
        records.append(dict(
            site=landing["idx"], speed=speed, soil=mode, pocket_pad=pocket_pad, slope_deg=round(landing["slope_deg"], 2),
            pocket_pad_max_sinkage_cm=100 * pads[pocket_pad]["max_sinkage_m"],
            other_pads_max_sinkage_cm=100 * max(pads[k]["max_sinkage_m"] for k in others),
            other_pads_mean_max_sinkage_cm=100 * float(np.mean([pads[k]["max_sinkage_m"] for k in others])),
            differential_sinkage_cm=100 * (max(p["max_sinkage_m"] for p in pads.values()) - min(p["max_sinkage_m"] for p in pads.values())),
            pocket_pad_peak_force_kN=pads[pocket_pad]["peak_normal_force_N"] / 1000, pocket_pad_final_share=pads[pocket_pad]["final_load_share"],
            pocket_pad_soil_stiffness_MPa_per_m=pads[pocket_pad]["soil_tangent_modulus_at_5cm_Pa_per_m"] / 1e6,
            final_tilt_deg=landing["tilt_deg"], peak_tilt_deg=landing["peak_tilt_deg"], rest_s=landing["rest_s"],
            outcome=landing["outcome"], touching=" ".join(landing["touching"]), body_contact=landing["body_contact"],
            two_pad_line_distance_cm=None if landing.get("two_pad_line_distance_m") is None else 100 * landing["two_pad_line_distance_m"]))
    return records


def figure(records, path):
    """Pocket-pad sinkage vs the other pads, and tilt, heterogeneous vs uniform, per site and speed."""
    sites = sorted({r["site"] for r in records})
    speeds = sorted({r["speed"] for r in records})
    by_key = {(r["soil"], r["site"], r["speed"]): r for r in records}
    panels = (("pocket_pad_max_sinkage_cm", "Max sinkage of the pad over the pocket (cm)"),
              ("other_pads_max_sinkage_cm", "Max sinkage of the other three pads (cm)"),
              ("peak_tilt_deg", "Peak tilt (deg)"), ("final_tilt_deg", "Final tilt (deg)"))
    figure_handle, axes = plt.subplots(2, 2, figsize=(14, 8.2), constrained_layout=True)
    axes = axes.ravel()
    labels = [f"{site}\n{speed:g} m/s" for speed in speeds for site in sites]
    x = np.arange(len(labels))
    for axis, (key, title) in zip(axes, panels):
        for offset, mode in ((-0.2, "uniform"), (0.2, "heterogeneous")):
            values = [by_key.get((mode, site, speed), {}).get(key, np.nan) for speed in speeds for site in sites]
            axis.bar(x + offset, values, 0.38, color=MODE_COLORS[mode], label=f"{mode} soil")
        axis.set_xticks(x, labels, fontsize=7.5)
        axis.set_title(title, fontsize=9.5, color=INK)
        axis.grid(True, axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes[2:]:
        axis.set_xlabel("site, touchdown speed")
    figure_handle.suptitle("Apollo LM on deformable regolith (pressure-sinkage + shear per footpad): sites with ONE pad over a soft pocket.\n"
                           "uniform = same soil profile everywhere, no pocket.  SYNTHETIC soil fields; lunar Bekker constants (literature)", fontsize=10.5)
    figure_handle.savefig(path, dpi=130)
    plt.close(figure_handle)


def main():
    """Summary files, figure and a printed table."""
    candidates = json.loads(SITE_FILE.read_text())["candidates"]
    pocket_pad_of_site = {c["site_index"]: c["pocket_pad"] for c in candidates}
    records = summarise(load(), pocket_pad_of_site)
    if not records:
        raise SystemExit("no results/phase2/soil_touchdown/*.json - run run_phase2.sh soil-touchdown first")
    summary_directory = study_paths.phase_results_directory("phase2")
    (summary_directory / "soil_touchdown_summary.json").write_text(json.dumps(dict(records=records, provenance=study_paths.git_provenance()), indent=1))
    with open(summary_directory / "soil_touchdown_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    figure(records, study_paths.FIGURES_DIRECTORY / "soil_touchdown_comparison.png")
    print(f"{'site':>4s} {'v':>3s} {'soil':13s} {'pocket pad':>10s} {'pocket cm':>9s} {'others cm':>9s} {'diff cm':>7s} {'tilt':>5s} "
          f"{'peak':>5s} {'share':>5s} {'outcome':8s} touching")
    for r in sorted(records, key=lambda r: (r["site"], r["speed"], r["soil"])):
        print(f"{r['site']:4d} {r['speed']:3g} {r['soil']:13s} {r['pocket_pad']:>10s} {r['pocket_pad_max_sinkage_cm']:9.1f} "
              f"{r['other_pads_max_sinkage_cm']:9.1f} {r['differential_sinkage_cm']:7.1f} {r['final_tilt_deg']:5.2f} {r['peak_tilt_deg']:5.2f} "
              f"{r['pocket_pad_final_share']:5.2f} {r['outcome']:8s} {r['touching']}"
              + (f"   (COM {r['two_pad_line_distance_cm']:.1f} cm from the two-pad line)" if r["two_pad_line_distance_cm"] is not None else ""))


if __name__ == "__main__":
    main()
