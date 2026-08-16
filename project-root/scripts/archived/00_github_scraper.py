import requests
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project-root
RAW_DIR = BASE_DIR / "data" / "raw"

GITHUB_OWNER = "martj42"
GITHUB_REPO = "international_results"
GITHUB_BRANCH = "master"  # change to "main" if needed

FILES = ["shootouts.csv", "goalscorers.csv", "former_names.csv"]

# -----------------------
# DOWNLOAD
# -----------------------
def download_files(out_dir: str | Path, branch: str = GITHUB_BRANCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{branch}"
    headers = {"User-Agent": "Mozilla/5.0"}

    saved_paths: list[Path] = []

    for filename in FILES:
        url = f"{base_url}/{filename}"
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()

        out_path = out_dir / filename
        out_path.write_bytes(resp.content)

        print(f"✅ Downloaded {filename} -> {out_path}")
        saved_paths.append(out_path)

    return saved_paths

# -----------------------
# PIPELINE
# -----------------------
def main():
    download_files(RAW_DIR)

if __name__ == "__main__":
    main()