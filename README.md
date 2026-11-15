# Lunar touchdown on heterogeneous regolith

**Question.** Can a lunar lander touch down safely when its footpads meet different soil conditions? Imagine three
pads landing on firm ground while the fourth meets a soft subsurface pocket. That pad sinks further, which causes
differential settlement, shifts the load distribution, and can produce a dangerous roll or pitch.

The study uses the **Apollo 17 landing site** (real LROC NAC terrain), the **Apollo Lunar Module** (NASA mass
properties) in **NVIDIA Isaac Sim 6 / PhysX**, and **synthetic, spatially correlated regolith** generated on the same
terrain.

> All soil properties and friction values in this study are **synthetic and uncalibrated**. They come from procedural
> soil fields, not measurements. Only the terrain surface (LROC NAC DTM) is observed data. Results compare scenarios;
> they are not predictions for the real Taurus-Littrow soil.

## Architecture

```
NAC terrain -> heterogeneous subsurface -> general soil-contact model -> lander touchdown -> rover mobility -> autonomous operations
 (lunar_regolith)   (lunar_regolith)           (lunar_regolith.contact)    (THIS STUDY)          (future)             (future)
```

| Component | Repository | Role here |
|---|---|---|
| `external/lunar_regolith` | [MEO41/lunar_regolith](https://github.com/MEO41/lunar_regolith) | Synthetic soil fields on the DTM. Also the **general soil-contact model**: patch-averaged soil under a contact, interface friction, USD friction zones. Rovers will reuse it later. |
| `external/apollo_lander_modeling` | [MEO41/isaac-lunar-terrain-setup-apollo](https://github.com/MEO41/isaac-lunar-terrain-setup-apollo) | Apollo LM and the Apollo 17 scene in Isaac Sim: exact terrain colliders, lander mass properties, Isaac launcher. |
| this repository | MEO41/lunar_touchdown_study | Experiments, run scripts, analysis and write-up. |

Both dependencies are **git submodules pinned to exact commits**, and every result file records the commits of all
three repositories that produced it.

## Phases

| Phase | Content | Status |
|---|---|---|
| **1. Static heterogeneous terrain** | NAC terrain as a USD collider with **spatially varying friction materials** from the soil fields. Footpad contacts are linked to soil properties. Touchdowns on real slopes and uneven ground, measuring stability and footpad load distribution. | in progress |
| 2. Deformable regolith | Per-footpad soil queries, pressure-sinkage, differential penetration, shear and lateral displacement, terrain deformation, soft pockets and depth-dependent layers, coupled to the lander's 6-DoF dynamics. | next |
| 3. Uncertain terrain, risk-aware control | Estimating soil before touchdown, landing-site suitability, predicting differential settlement, trajectories and touchdown speeds, control robust to uncertain soil, partial or asymmetric contact. | later |

## Setup

```bash
git clone --recurse-submodules https://github.com/MEO41/lunar_touchdown_study.git && cd lunar_touchdown_study
# lunar_regolith: its own venv (soil generation, contact model, analysis)
python3 -m venv external/lunar_regolith/.venv && external/lunar_regolith/.venv/bin/pip install -e "external/lunar_regolith[test,usd]"
# apollo_lander_modeling: Isaac Sim 6 environment + data + scene (see its README): .env.local (ISAAC_ENV), ./download_data.sh, ./build_scene.sh
```

To use existing working copies that already hold the data, copy `.env.example` to `.env.local` and set
`LANDER_REPOSITORY` and `REGOLITH_REPOSITORY`.

## Phase 1: running it

```bash
experiments/phase1_static_friction/run_phase1.sh all       # or step by step:
experiments/phase1_static_friction/run_phase1.sh soil      # soil maps on the lander DTM + friction overlay for the Near2m collider
experiments/phase1_static_friction/run_phase1.sh gate      # PhysX gate test of per-face friction zones
experiments/phase1_static_friction/run_phase1.sh touchdown 1 2 3   # 300 sites x {uniform, heterogeneous} x speeds
experiments/phase1_static_friction/run_phase1.sh annotate  # soil state under every footpad
experiments/phase1_static_friction/run_phase1.sh plot      # comparison tables and figures
```

**Phase 1 method**
- **Terrain:** the lander scene's `Near2m` tier (1.2 km, 2 m DTM, exact triangle-mesh collider, rigid).
- **Friction:** `lunar_regolith contact-overlay` writes a USD layer assigning **8 friction zones** to its 720,000 faces,
  with μ<sub>s</sub> 0.46–1.10 and an area median of 0.80. Uniform runs use 0.80 / 0.70 everywhere, so the two modes
  differ only in spatial variation.
- **Lander:** the footpads carry a μ = 1 contact material, and the terrain materials use PhysX's `multiply` combine
  mode, so each pad feels exactly the zone value. The gate test verifies this.
- **Touchdowns:** at an exact vertical speed (1, 2 and 3 m/s) on 300 Near2m sites.
- **Recorded:** per-pad contact order, load-share history, slip, the friction zone under each pad, peak and final tilt,
  the minimum support margin of the centre of mass, the outcome class, and the patch-averaged soil under each pad.
  The soil record includes soft-pocket coverage, flagged for Phase 2.

### Phase 1 findings so far

- **The PhysX gate test passes 12/12.** Per-face friction zones are honoured. The measured dynamic friction matches the
  zone values within 0.001 (0.406 and 0.966).
- **The earlier drop-test baseline ran with less friction than intended.** The lander in `apollo_lander_modeling` had
  no physics material, so PhysX averaged its default 0.5 with the regolith's 0.8 / 0.7. The effective friction was
  **0.65 / 0.60**, not 0.80 / 0.70. All runs in this study set the footpad material explicitly.
- Touchdown results: pending (the runs are in progress).

## Layout

```
common/study_paths.py                     paths (.env.local / submodules) and git provenance
experiments/phase1_static_friction/       run_phase1.sh, verify_friction_zones_in_physx.py (Isaac), heterogeneous_touchdown_test.py (Isaac),
                                          annotate_footpad_soil.py, plot_heterogeneous_touchdown.py
external/                                 pinned submodules: lunar_regolith, apollo_lander_modeling
results/                                  run outputs (git-ignored)        docs/images/   curated figures
```
