#!/usr/bin/env python3
"""Chrono SCM vs Isaac Sim (our per-footpad coupling) for the 30 Phase 2 touchdowns (plain Python; Phase 2 cross-check).

    experiments/phase2_crosscheck/run_crosscheck.sh compare

Pairs results/phase2/crosscheck/chrono_<mode>_v<speed>.json with results/phase2/soil_touchdown/soil_touchdown_<mode>_v<speed>.json landing
by landing and writes results/phase2/crosscheck/chrono_vs_isaac.json / .csv and docs/images/chrono_vs_isaac.png (pad sinkage, tilt and the
pocket effect - heterogeneous minus uniform sinkage of the pocket pad - in both engines).
Known differences: SCM is single-layer Bekker per node (our layered curve linearised at 5 cm), SCM deforms its grid and samples the
terrain mesh at 0.1 m, footpads are separate 50 kg bodies on spherical joints in Chrono vs force points on the rigid lander in Isaac.
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

CHRONO = study_paths.RESULTS_ROOT / "phase2" / "crosscheck"
ISAAC = study_paths.RESULTS_ROOT / "phase2" / "soil_touchdown"
SITE_FILE = study_paths.RESULTS_ROOT / "phase2" / "soft_pocket_demo" / "soft_pocket_site.json"
PADS = ("+X", "+Y", "-X", "-Y")
INK, MUTED, SURFACE, GRID = "#1a1a19", "#5f5e5a", "#fcfcfb", "#d9d8d4"
POCKET, OTHER = "#eb6834", "#2a78d6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def results(directory, prefix):
    """{(mode, speed, site): landing} for every result file of one engine."""
    found = {}
    for path in directory.glob(f"{prefix}_*_v*.json"):
        match = re.fullmatch(rf"{prefix}_(heterogeneous|uniform)_v([0-9.]+)\.json", path.name)
        if match:
            for landing in json.loads(path.read_text())["results"]:
                found[(match.group(1), float(match.group(2)), landing["idx"])] = landing
    return found


QUASI_STATIC_CREEP_M_S, QUASI_STATIC_TILT_DEG_S = 0.001, 0.01


def chrono_outcome(landing):
    """Chrono outcome; a landing still 'moving' only by slow SCM creep (< 1 mm/s, tilt drift < 0.01 deg/s over the last 5 s) is
    classified from its quasi-static state with the same rule as Isaac."""
    creep, drift = landing.get("creep_speed_last_5s_m_per_s"), landing.get("tilt_drift_last_5s_deg_per_s")
    if landing["outcome"] != "moving" or creep is None or creep > QUASI_STATIC_CREEP_M_S or drift > QUASI_STATIC_TILT_DEG_S:
        return landing["outcome"]
    margin = landing["com_margin_m"]
    upright = len(landing["touching"]) >= 3 and margin == margin and margin > 0 and landing["tilt_deg"] <= landing["slope_deg"] + 5
    return "upright (quasi-static)" if upright else "tilted (quasi-static)"


def main():
    """Pair, tabulate and plot."""
    pocket_pad = {c["site_index"]: c["pocket_pad"] for c in json.loads(SITE_FILE.read_text())["candidates"]}
    chrono, isaac = results(CHRONO, "chrono"), results(ISAAC, "soil_touchdown")
    rows = []
    for key in sorted(set(chrono) & set(isaac)):
        mode, speed, site = key
        a, b = isaac[key], chrono[key]
        for pad in PADS:
            rows.append(dict(site=site, speed=speed, soil=mode, pad=pad, is_pocket_pad=pad == pocket_pad[site],
                             isaac_max_sinkage_cm=100 * a["pads"][pad]["max_sinkage_m"], chrono_max_sinkage_cm=100 * b["pads"][pad]["max_sinkage_m"],
                             isaac_final_share=a["pads"][pad]["final_load_share"], chrono_final_share=b["pads"][pad]["final_load_share"],
                             isaac_tilt_deg=a["tilt_deg"], chrono_tilt_deg=b["tilt_deg"], isaac_outcome=a["outcome"], chrono_outcome=chrono_outcome(b),
                             chrono_creep_mm_s=(1000 * b["creep_speed_last_5s_m_per_s"] if b["outcome"] == "moving"
                                                and b.get("creep_speed_last_5s_m_per_s") is not None else None),   # meaningful for 30 s runs only
                             chrono_step_s=b.get("step_s")))
    if not rows:
        raise SystemExit("no paired results - run run_crosscheck.sh touchdown and run_phase2.sh soil-touchdown first")
    landings = {(r["soil"], r["speed"], r["site"]): r for r in rows}
    isaac_sink = np.array([r["isaac_max_sinkage_cm"] for r in rows])
    chrono_sink = np.array([r["chrono_max_sinkage_cm"] for r in rows])
    tilt_pairs = np.array([(r["isaac_tilt_deg"], r["chrono_tilt_deg"]) for r in landings.values()])
    effect = []
    for site in sorted(pocket_pad):
        for speed in sorted({r["speed"] for r in rows}):
            get = {(r["soil"], r["engine"]): r["value"] for r in [
                dict(soil=row["soil"], engine=engine, value=row[f"{engine}_max_sinkage_cm"])
                for row in rows if row["site"] == site and row["speed"] == speed and row["is_pocket_pad"] for engine in ("isaac", "chrono")]}
            if len(get) == 4:
                effect.append(dict(site=site, speed=speed, isaac_pocket_effect_cm=get[("heterogeneous", "isaac")] - get[("uniform", "isaac")],
                                   chrono_pocket_effect_cm=get[("heterogeneous", "chrono")] - get[("uniform", "chrono")]))
    summary = dict(
        pad_sinkage_correlation=float(np.corrcoef(isaac_sink, chrono_sink)[0, 1]),
        pad_sinkage_median_difference_cm=float(np.median(chrono_sink - isaac_sink)),
        pad_sinkage_median_absolute_difference_cm=float(np.median(np.abs(chrono_sink - isaac_sink))),
        tilt_median_absolute_difference_deg=float(np.median(np.abs(tilt_pairs[:, 1] - tilt_pairs[:, 0]))),
        same_outcome=int(sum(r["isaac_outcome"] == r["chrono_outcome"].split(" ")[0] for r in landings.values())), landings=len(landings),
        chrono_quasi_static=int(sum("quasi-static" in r["chrono_outcome"] for r in landings.values())),
        pocket_effect_correlation=float(np.corrcoef([e["isaac_pocket_effect_cm"] for e in effect], [e["chrono_pocket_effect_cm"] for e in effect])[0, 1]) if len(effect) > 2 else None,
        pocket_effect=effect, provenance=study_paths.git_provenance())
    (CHRONO / "chrono_vs_isaac.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    with open(CHRONO / "chrono_vs_isaac.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    pocket = np.array([r["is_pocket_pad"] for r in rows])
    limit = max(isaac_sink.max(), chrono_sink.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
    axes[0].scatter(isaac_sink[~pocket], chrono_sink[~pocket], s=16, color=OTHER, label="other pads")
    axes[0].scatter(isaac_sink[pocket], chrono_sink[pocket], s=22, color=POCKET, label="pad over the soft pocket")
    axes[0].set_xlabel("Isaac Sim + our coupling: max pad sinkage (cm)")
    axes[0].set_ylabel("Chrono SCM: max pad sinkage (cm)")
    axes[0].set_title(f"Pad sinkage, {len(rows)} pads (r = {summary['pad_sinkage_correlation']:.2f})", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    limit = tilt_pairs.max() * 1.1
    axes[1].plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
    axes[1].scatter(tilt_pairs[:, 0], tilt_pairs[:, 1], s=18, color=INK)
    axes[1].set_xlabel("Isaac: final tilt (deg)")
    axes[1].set_ylabel("Chrono: final tilt (deg)")
    creeping = sum(1 for r in landings.values() if r["chrono_outcome"] == "moving" and r["isaac_outcome"] != "moving")
    axes[1].set_title(f"Final tilt, {len(landings)} landings; same outcome {summary['same_outcome']}/{summary['landings']}\n"
                      f"({creeping} others: Chrono lander still creeping downslope, SCM shear)", fontsize=10)
    if effect:
        x = np.array([e["isaac_pocket_effect_cm"] for e in effect])
        y = np.array([e["chrono_pocket_effect_cm"] for e in effect])
        limit = max(x.max(), y.max()) * 1.1
        axes[2].plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
        axes[2].scatter(x, y, s=22, color=POCKET)
        for e in effect:
            axes[2].annotate(f"{e['site']}/{e['speed']:g}", (e["isaac_pocket_effect_cm"], e["chrono_pocket_effect_cm"]), fontsize=7, color=MUTED,
                             xytext=(3, 3), textcoords="offset points")
        axes[2].set_xlabel("Isaac: pocket effect (cm)")
        axes[2].set_ylabel("Chrono: pocket effect (cm)")
        axes[2].set_title("Pocket effect = pocket-pad sinkage, heterogeneous - uniform", fontsize=10)
    for axis in axes:
        axis.grid(True, color=GRID, lw=0.5)
    figure.suptitle("Cross-check: Project Chrono SCM vs Isaac Sim with lunar_regolith's pressure-sinkage coupling (dashed: equal).  "
                    "Same terrain, soil maps, lander and release.  SYNTHETIC soil", fontsize=10.5)
    figure.savefig(study_paths.FIGURES_DIRECTORY / "chrono_vs_isaac.png", dpi=130)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pocket_effect", "provenance")}, indent=1))


if __name__ == "__main__":
    main()
