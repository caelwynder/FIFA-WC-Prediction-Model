import requests
from pathlib import Path

def download_results_csv(
    out_path: str | Path,
    branch: str = "master",
):
    """
    Downloads results.csv from martj42/international_results on GitHub and saves it locally.

    Repo: martj42/international_results
    File: results.csv
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://raw.githubusercontent.com/martj42/international_results/{branch}/results.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    print(f"✅ Downloaded results.csv -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: save into your project data/raw folder
    BASE_DIR = Path(__file__).resolve().parent.parent  # project-root
    save_to = BASE_DIR / "data" / "raw" / "results.csv"
    download_results_csv(save_to)
