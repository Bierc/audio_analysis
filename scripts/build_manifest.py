from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data"

rows = []
# include single-note set
for p in sorted((DATA / "00_single_sample").glob("*.wav")):
    rows.append({"split": "single", "instrument": p.stem, "path": str(p)})

# include melodic sets
for inst_dir in ["bass", "flute", "piano", "trumpet"]:
    for p in sorted((DATA / inst_dir).glob("*.wav")):
        rows.append({"split": "melody", "instrument": inst_dir, "path": str(p)})

df = pd.DataFrame(rows)
out = ROOT / "outputs" / "manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved manifest with {len(df)} rows -> {out}")
