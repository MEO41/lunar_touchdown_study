# CLAUDE.md — lunar_touchdown_study

Research project: **can a lunar lander touch down safely on spatially heterogeneous regolith, where its footpads meet
different soil conditions?** (Apollo 17 site, Apollo LM, Isaac Sim / PhysX.)

## Git workflow (user instruction)

- **Commit and push as you go** (origin = https://github.com/MEO41/lunar_touchdown_study, branch `main`) at sensible
  checkpoints, so progress can be followed commit by commit. State in the message if something is knowingly incomplete,
  and which tests or runs were actually executed.
- Never commit results (`results/` is git-ignored), DTMs, generated terrain or `.env.local`. Curated figures go in `docs/images/`.
- After changing a dependency repository, push it first and then update its pin here
  (`git -C external/<repo> pull` or `checkout <commit>`, then commit the new submodule pointer).

## Architecture (user instruction)

NAC terrain → heterogeneous subsurface → **general soil-contact model** → lander touchdown → rover mobility → autonomous operations.

| Where | What belongs there |
|---|---|
| `external/lunar_regolith` (MEO41/lunar_regolith) | Terrain/soil generation and the GENERAL soil-contact model (`lunar_regolith.contact`): anything a rover could reuse, including Phase 2 pressure-sinkage/shear. |
| `external/apollo_lander_modeling` (MEO41/isaac-lunar-terrain-setup-apollo) | Apollo LM + Apollo 17 scene for Isaac Sim: lander/scene assets and fixes only. Its own CLAUDE.md applies there (plan first, verbose names, docstrings). |
| this repository | Study-specific experiments, run scripts, analysis, figures, write-up. No reusable soil mechanics here. |

Do not build lander- or rover-specific terramechanics inside lunar_regolith, and do not copy code between repositories.

## Conventions

- Paths only through `common/study_paths.py` (environment variable > `.env.local` > submodule). On this machine `.env.local`
  points at the working copies in `~/Desktop/MEO/` (their DTMs, Isaac scene and soil run are already built).
- Every result file records `study_paths.git_provenance()` (commits of all three repositories).
- Isaac scripts run through the lander repository's `isaac_run.sh` (Isaac Sim 6 Python, see its `.env.local`); they may
  import only numpy / scipy / pxr / Isaac modules plus `common/study_paths.py` and `lunar_regolith.contact.{friction_zones,usd_terrain}`.
  Analysis scripts run with the lunar_regolith Python (`REGOLITH_PYTHON`). Project Chrono scripts (`experiments/phase2_crosscheck/chrono_*.py`)
  run with `CHRONO_PYTHON` (PyChrono 10 from the conda channel `projectchrono`; the conda-forge build lacks the vehicle/SCM module) and may
  import only numpy, pychrono and `common/`.
- Descriptive names (no cryptic abbreviations) and a docstring on every function, as in the lander repository.
- All soil and friction values are SYNTHETIC / uncalibrated unless calibrated against measurements; label them so in
  figures, metadata and text. The NAC DTM surface is observed data and must never be modified.
- Report only results of runs and tests that were actually executed.
