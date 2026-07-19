#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="Exp271431"
FeatureConfig="ConfigC"
ModelName="EnhancedUNet"
RunName="Bs4Ep40_D010_LR75e5_UnsupervisedWindMBMPPlus"
Split="Test"
Checkpoint="BestModel.pt"

echo ""
echo "============================================================"
echo "TRAIN ConfigC EnhancedUNet V2"
echo "============================================================"

python Scripts/Step10TrainSegmentationModelClean.py \
  --RunTag "$RunTag" \
  --FeatureConfig "$FeatureConfig" \
  --ModelName "$ModelName" \
  --RunName "$RunName" \
  --Epochs 40 \
  --BatchSize 4 \
  --LearningRate 7.5e-5 \
  --WeightDecay 1e-5 \
  --Dropout 0.10 \
  --BaseChannels 32 \
  --DiceWeight 1.0 \
  --Threshold 0.5

echo ""
echo "============================================================"
echo "EVALUATE ConfigC EnhancedUNet V2 at multiple thresholds"
echo "============================================================"

for Threshold in 0.40 0.50 0.60 0.70; do
  echo ""
  echo "---- Threshold ${Threshold} ----"

  python Scripts/Step11EvaluateSegmentationModelClean.py \
    --RunTag "$RunTag" \
    --FeatureConfig "$FeatureConfig" \
    --ModelName "$ModelName" \
    --RunName "$RunName" \
    --Split "$Split" \
    --Checkpoint "$Checkpoint" \
    --BatchSize 8 \
    --Threshold "$Threshold"

  mkdir -p "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Metrics/Threshold_${Threshold}"
  cp \
    "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Metrics/${Split}MetricsSummary.csv" \
    "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Metrics/Threshold_${Threshold}/${Split}MetricsSummary.csv"

  cp \
    "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Metrics/${Split}MetricsBySample.csv" \
    "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Metrics/Threshold_${Threshold}/${Split}MetricsBySample.csv"
done

echo ""
echo "============================================================"
echo "BUILD THRESHOLD COMPARISON TABLE"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

RunTag = "Exp271431"
FeatureConfig = "ConfigC"
ModelName = "EnhancedUNet"
RunName = "Bs4Ep40_D010_LR75e5_UnsupervisedWindMBMPPlus"
Split = "Test"

root = Path("Outputs/Experiments") / RunTag / FeatureConfig / f"{ModelName}_{RunName}"
rows = []

 for_dir = sorted((root / "Metrics").glob("Threshold_*"))
for d in for_dir:
    threshold = d.name.replace("Threshold_", "")
    path = d / f"{Split}MetricsSummary.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    df.insert(0, "EvalThreshold", float(threshold))
    rows.append(df)

if not rows:
    raise SystemExit("No threshold summaries found.")

out = pd.concat(rows, ignore_index=True)
cols = [
    "EvalThreshold",
    "FeatureConfig",
    "ModelName",
    "RunName",
    "Split",
    "Samples",
    "InputChannels",
    "MeanDice",
    "MeanIoU",
    "MeanPrecision",
    "MeanRecall",
    "GlobalDice",
    "GlobalIoU",
    "GlobalPrecision",
    "GlobalRecall",
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
PY

echo ""
echo "============================================================"
echo "SELECT CASES AND VISUALIZE BEST THRESHOLD DEFAULT 0.50"
echo "============================================================"

python Scripts/Step12SelectVisualizationCases.py \
  --RunTag "$RunTag" \
  --FeatureConfig "$FeatureConfig" \
  --ModelName "$ModelName" \
  --RunName "$RunName" \
  --Split "$Split" \
  --FixedCount 9 \
  --BestCount 3 \
  --WorstCount 3 \
  --ErrorCount 3

python Scripts/Step13VisualizePredictions.py \
  --RunTag "$RunTag" \
  --FeatureConfig "$FeatureConfig" \
  --ModelName "$ModelName" \
  --RunName "$RunName" \
  --Split "$Split" \
  --Checkpoint "$Checkpoint" \
  --Threshold 0.5

echo ""
echo "DONE."
echo "Results:"
echo "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/"
