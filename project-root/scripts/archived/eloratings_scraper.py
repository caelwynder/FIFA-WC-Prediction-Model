import time
import requests
from pathlib import Path

BASE = "https://www.eloratings.net"
YEARS = range(2000, 2027)

# where to save the yearly .tsv files
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "tsv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/tab-separated-values",
    "Referer": "https://www.eloratings.net/"
})

for year in YEARS:
    url = f"{BASE}/{year}_results.tsv"
    out_path = OUT_DIR / f"{year}_results.tsv"

    if out_path.exists():
        print(f"Skipping {year} (already downloaded)")
        continue

    print(f"Downloading {year} -> {out_path}")
    r = session.get(url, timeout=60)
    r.raise_for_status()

    out_path.write_bytes(r.content)

    time.sleep(0.3)

print(f"\n✅ Done. Files saved in: {OUT_DIR}")
