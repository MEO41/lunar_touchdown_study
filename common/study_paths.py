"""Where everything lives, and which code versions produced a result (standard library only).

Imported by every study script - including scripts that run inside Isaac Sim's Python - so it must not need third-party packages.

Repository locations (first match wins):
    environment variable  >  .env.local in the study root (KEY=VALUE lines, git-ignored)  >  the pinned submodule under external/
    LANDER_REPOSITORY      Apollo LM + Apollo 17 scene for Isaac Sim  (default external/apollo_lander_modeling)
    REGOLITH_REPOSITORY    terrain/soil generator + general soil-contact model  (default external/lunar_regolith)
    REGOLITH_PYTHON        Python with lunar_regolith installed  (default <REGOLITH_REPOSITORY>/.venv/bin/python)
    REGOLITH_CONFIG        soil configuration, relative to REGOLITH_REPOSITORY  (default config/apollo17_lander_site.yaml)
    REGOLITH_RUN_NAME      generated soil run under <REGOLITH_REPOSITORY>/outputs  (default apollo17_lander_site)
    CHRONO_PYTHON          Python with PyChrono (Project Chrono, conda channel projectchrono) for the Phase 2 cross-check  (default: none)
The submodules pin exact commits; working copies elsewhere (with their large data already built) can be used through .env.local.
Every result records git_provenance(), so it is always known which commits produced it.
"""
import os
import subprocess
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]


def local_settings():
    """KEY=VALUE pairs from <study root>/.env.local (comments and blank lines ignored)."""
    settings = {}
    path = STUDY_ROOT / ".env.local"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = os.path.expanduser(value.strip().strip('"').strip("'"))
    return settings


def setting(name, default):
    """Value of a setting: environment variable, then .env.local, then `default`."""
    return os.environ.get(name) or local_settings().get(name) or default


LANDER_REPOSITORY = Path(setting("LANDER_REPOSITORY", STUDY_ROOT / "external" / "apollo_lander_modeling")).resolve()
REGOLITH_REPOSITORY = Path(setting("REGOLITH_REPOSITORY", STUDY_ROOT / "external" / "lunar_regolith")).resolve()
REGOLITH_PYTHON = Path(setting("REGOLITH_PYTHON", REGOLITH_REPOSITORY / ".venv" / "bin" / "python"))
REGOLITH_CONFIG = setting("REGOLITH_CONFIG", "config/apollo17_lander_site.yaml")
REGOLITH_RUN_NAME = setting("REGOLITH_RUN_NAME", "apollo17_lander_site")
REGOLITH_RUN_DIRECTORY = REGOLITH_REPOSITORY / "outputs" / REGOLITH_RUN_NAME
CHRONO_PYTHON = setting("CHRONO_PYTHON", "")

LANDER_SCENE = LANDER_REPOSITORY / "scene" / "apollo17_scene.usd"
LANDER_USD = LANDER_REPOSITORY / "usd" / "apollo_lunar_module.usd"
LANDER_NEAR_SITES = LANDER_REPOSITORY / "scene" / "drop_points_near.json"
LANDER_ISAAC_RUNNER = LANDER_REPOSITORY / "isaac_run.sh"

RESULTS_ROOT = STUDY_ROOT / "results"
FIGURES_DIRECTORY = STUDY_ROOT / "docs" / "images"


def phase_results_directory(phase_name, *parts):
    """results/<phase_name>/<parts...> (created on demand; results/ is git-ignored)."""
    path = RESULTS_ROOT.joinpath(phase_name, *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_state(repository):
    """Commit, dirty flag and origin URL of a git working copy (None values if it is not a git repository)."""
    def run(*arguments):
        """Run a git command in `repository`; return its stripped output or None."""
        try:
            return subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"path": str(repository), "commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "origin": run("remote", "get-url", "origin")}


def git_provenance():
    """Code versions behind a result: the study repository, the lander repository and the regolith repository."""
    return {"study": git_state(STUDY_ROOT), "lander_repository": git_state(LANDER_REPOSITORY),
            "regolith_repository": git_state(REGOLITH_REPOSITORY)}


if __name__ == "__main__":
    # shell form, for run scripts:  eval "$(python3 common/study_paths.py)"
    for name, value in (("STUDY_ROOT", STUDY_ROOT), ("LANDER_REPOSITORY", LANDER_REPOSITORY), ("REGOLITH_REPOSITORY", REGOLITH_REPOSITORY),
                        ("REGOLITH_PYTHON", REGOLITH_PYTHON), ("REGOLITH_CONFIG", REGOLITH_CONFIG), ("REGOLITH_RUN_NAME", REGOLITH_RUN_NAME),
                        ("REGOLITH_RUN_DIRECTORY", REGOLITH_RUN_DIRECTORY), ("LANDER_SCENE", LANDER_SCENE),
                        ("LANDER_ISAAC_RUNNER", LANDER_ISAAC_RUNNER), ("RESULTS_ROOT", RESULTS_ROOT), ("CHRONO_PYTHON", CHRONO_PYTHON)):
        print(f"{name}='{value}'")
