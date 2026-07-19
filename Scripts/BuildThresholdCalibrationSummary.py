#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--RunTag", required=True)
parser.add_argument("--FeatureConfig", required=True)
parser.add_argument("--ModelName", required=True)
parser.add_argument("--RunName", required=True)
parser.add_argument("--Split", default="Test")
args = parser.parse_args()

root = Path("Outputs/Experiments") / args.RunTag / args.FeatureConfig / f"{args.ModelName}_{args.RunName}"

rows = []
for d in sorted((root / "Metrics").glob("Threshold_*")):
    threshold = d.name.replace("Threshold_", "")
    path = d / f"{args.Split}MetricsSummary.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    df.insert(0, "EvalThreshold", float(threshold))
    rows.append(df)

if not rows:
    raise SystemExit("No threshold summaries found.")

out = pd.concat(rows, ignore_index=True)

cols = [
    "EvalThreshold", "FeatureConfig", "ModelName", "RunName", "Split",
    "Samples", "InputChannels",
    "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall",
    "GlobalDice", "GlobalIoU", "GlobalPrecision", "GlobalRecall",
    "GroundTruthPixels", "PredictedPixels",
    "TruePositivePixels", "FalsePositivePixels", "FalseNegativePixels",
]
cols = [c for c in cols if c in out.columns]
out = out[cols].sort_values("EvalThreshold")

out_dir = root / "Tables"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "ThresholdCalibrationSummary.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("")
print("Saved:", out_path)
