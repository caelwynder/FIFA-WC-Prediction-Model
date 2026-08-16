# scripts/run_full_pipeline.py
#
# One-command refresh: scrape the latest results/Elo ratings, rebuild the
# model dataset with up-to-date rolling features, and retrain both models.
#
#   python scripts/run_full_pipeline.py
#
# Equivalent to running, in order:
#   00_run_preprocessing_pipeline.py  (scrape + merge -> data/processed/00_merged_dataset.csv)
#   01_build_model_dataset.py         (feature engineering -> data/processed/01_model_dataset.csv)
#   02_train_outcome_model.py         (train + save -> models/*.joblib)

import sys
import time
import platform
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project-root
SCRIPT_DIR = BASE_DIR / "scripts"

STEPS = [
    SCRIPT_DIR / "00_run_preprocessing_pipeline.py",
    SCRIPT_DIR / "01_build_model_dataset.py",
    SCRIPT_DIR / "02_train_outcome_model.py",
    SCRIPT_DIR / "05_train_penalty_model.py",
]


def build_env() -> dict:
    """
    xgboost's macOS wheel links against libomp via @rpath but expects it at
    /opt/homebrew/opt/libomp (a Homebrew path). On machines without Homebrew,
    import fails even though a copy of libomp.dylib already ships inside
    scikit-learn's bundled .dylibs. Point DYLD_LIBRARY_PATH at that copy so
    xgboost resolves it without requiring any extra system install.
    """
    import os

    env = os.environ.copy()

    if platform.system() != "Darwin":
        return env

    # Do NOT resolve() sys.executable: venv binaries are symlinks to the base
    # interpreter, and resolving would point us at the base install's
    # site-packages (where sklearn/xgboost aren't installed) instead of the venv's.
    dylibs_dir = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "sklearn" / ".dylibs"
    if not dylibs_dir.is_dir():
        return env

    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{dylibs_dir}:{existing}" if existing else str(dylibs_dir)
    return env


def run_step(script_path: Path, env: dict):
    print("\n" + "#" * 70)
    print(f"STEP: {script_path.name}")
    print("#" * 70)

    started = time.monotonic()
    subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), env=env, check=True)
    elapsed = time.monotonic() - started

    print(f"-- {script_path.name} finished in {elapsed:.1f}s --")


def main():
    env = build_env()
    started = time.monotonic()

    for step in STEPS:
        if not step.exists():
            raise FileNotFoundError(f"Pipeline step not found: {step}")
        run_step(step, env)

    elapsed = time.monotonic() - started
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("  - Scraped data merged   -> data/processed/00_merged_dataset.csv")
    print("  - Model dataset rebuilt -> data/processed/01_model_dataset.csv")
    print("  - Models retrained      -> models/rf_outcome_model.joblib, models/xgb_outcome_model.joblib")
    print("  - Penalty model trained -> models/penalty_model.joblib")
    print("=" * 70)


if __name__ == "__main__":
    main()
