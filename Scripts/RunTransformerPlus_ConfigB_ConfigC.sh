#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="${1:-$(date +%d%H%M)}"
SourceRunTag="${2:-101622}"

Split="Test"
Checkpoint="BestModel.pt"
TrainThreshold="0.50"
Thresholds=("0.30" "0.40" "0.50" "0.60" "0.70")

ModelName="TransformerUNet"
RunName="Bs4Ep50_TL4_TH4_SE_D010_LR5e5_TransformerPlus"

Epochs=50
BatchSize=4
LearningRate=5e-5
WeightDecay=1e-5
Dropout=0.10
BaseChannels=32
TransformerLayers=4
TransformerHeads=4
ReflectPadding=4

LogDir="Outputs/Experiments/${RunTag}/Logs"
mkdir -p "${LogDir}"

EchoLine() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

RequireFile() {
  local FilePath="$1"
  if [[ ! -f "$FilePath" ]]; then
    echo "ERROR: falta archivo requerido: $FilePath"
    exit 1
  fi
}

PrepareRunTag() {
  EchoLine "PREPARE RUN TAG ${RunTag}"

  mkdir -p "Outputs/Experiments/${RunTag}/Tables"

  for File in \
    SplitTrainFeatureReady.csv \
    SplitValidationFeatureReady.csv \
    SplitTestFeatureReady.csv
  do
    RequireFile "Outputs/Experiments/${SourceRunTag}/Tables/${File}"
    cp "Outputs/Experiments/${SourceRunTag}/Tables/${File}" \
       "Outputs/Experiments/${RunTag}/Tables/${File}"
    echo "Copied ${File}"
  done

  cat > "Outputs/Experiments/${RunTag}/RunMetadata_TransformerPlus.txt" <<EOF
RunTag=${RunTag}
SourceRunTag=${SourceRunTag}
CreatedAt=$(date -Iseconds)
Purpose=TransformerUNetPlus experiment for ConfigB and ConfigC.
ModelName=${ModelName}
RunName=${RunName}
Epochs=${Epochs}
BatchSize=${BatchSize}
LearningRate=${LearningRate}
Dropout=${Dropout}
BaseChannels=${BaseChannels}
TransformerLayers=${TransformerLayers}
TransformerHeads=${TransformerHeads}
UseSqueezeExcitation=True
ReflectPadding=${ReflectPadding}
Thresholds=${Thresholds[*]}
EOF
}

CompileCheck() {
  EchoLine "COMPILE CHECK"

  python -m py_compile Scripts/Step07BuildFeaturesClean.py
  python -m py_compile Scripts/Step08CheckFeatureTensorsClean.py
  python -m py_compile Scripts/Step10TrainSegmentationModelClean.py
  python -m py_compile Scripts/Step11EvaluateSegmentationModelClean.py
  python -m py_compile Scripts/Step12SelectVisualizationCases.py
  python -m py_compile Scripts/Step13VisualizePredictions.py

  echo "Compile OK."
}

BuildAndCheckFeatures() {
  local FeatureConfig="$1"

  EchoLine "BUILD FEATURES ${FeatureConfig}"

  python Scripts/Step07BuildFeaturesClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --UseFeatureReadySplits \
    --ClipValue 8.0

  EchoLine "CHECK FEATURES ${FeatureConfig}"

  python Scripts/Step08CheckFeatureTensorsClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}"
}

TrainTransformerPlus() {
  local FeatureConfig="$1"

  EchoLine "TRAIN TRANSFORMER PLUS | ${FeatureConfig}"

  python Scripts/Step10TrainSegmentationModelClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Epochs "${Epochs}" \
    --BatchSize "${BatchSize}" \
    --LearningRate "${LearningRate}" \
    --WeightDecay "${WeightDecay}" \
    --Dropout "${Dropout}" \
    --BaseChannels "${BaseChannels}" \
    --DiceWeight 1.0 \
    --Threshold "${TrainThreshold}" \
    --TransformerLayers "${TransformerLayers}" \
    --TransformerHeads "${TransformerHeads}" \
    --UseSqueezeExcitation \
    --ReflectPadding "${ReflectPadding}"
}

EvaluateThresholds() {
  local FeatureConfig="$1"
  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  EchoLine "EVALUATE THRESHOLDS | ${FeatureConfig}"

  for Threshold in "${Thresholds[@]}"; do
    echo ""
    echo "---- Threshold ${Threshold} ----"

    python Scripts/Step11EvaluateSegmentationModelClean.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --Split "${Split}" \
      --Checkpoint "${Checkpoint}" \
      --BatchSize 8 \
      --Threshold "${Threshold}"

    mkdir -p "${ModelRoot}/Metrics/Threshold_${Threshold}"

    cp "${ModelRoot}/Metrics/${Split}MetricsSummary.csv" \
       "${ModelRoot}/Metrics/Threshold_${Threshold}/${Split}MetricsSummary.csv"

    cp "${ModelRoot}/Metrics/${Split}MetricsBySample.csv" \
       "${ModelRoot}/Metrics/Threshold_${Threshold}/${Split}MetricsBySample.csv"
  done
}

BuildThresholdSummaryAndPromoteBest() {
  local FeatureConfig="$1"
  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  EchoLine "BUILD THRESHOLD SUMMARY AND PROMOTE BEST | ${FeatureConfig}"

  python - <<PY
from pathlib import Path
import pandas as pd
import json

RunTag = "${RunTag}"
FeatureConfig = "${FeatureConfig}"
ModelName = "${ModelName}"
RunName = "${RunName}"
Split = "${Split}"

root = Path("Outputs/Experiments") / RunTag / FeatureConfig / f"{ModelName}_{RunName}"

rows = []

for d in sorted((root / "Metrics").glob("Threshold_*")):
    threshold_text = d.name.replace("Threshold_", "")
    path = d / f"{Split}MetricsSummary.csv"

    if not path.exists():
        continue

    df = pd.read_csv(path)
    df.insert(0, "EvalThreshold", float(threshold_text))
    rows.append(df)

if not rows:
    raise SystemExit(f"No threshold summaries found for {root}")

out = pd.concat(rows, ignore_index=True)

cols = [
    "EvalThreshold",
    "FeatureConfig",
    "ModelName",
    "RunName",
    "RunId",
    "ModelRunId",
    "Split",
    "Checkpoint",
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
    "GroundTruthPixels",
    "PredictedPixels",
    "TruePositivePixels",
    "FalsePositivePixels",
    "FalseNegativePixels",
]
cols = [c for c in cols if c in out.columns]
out = out[cols].sort_values("EvalThreshold")

out_dir = root / "Tables"
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "ThresholdCalibrationSummary.csv"
out.to_csv(summary_path, index=False)

rank_cols = [c for c in ["MeanDice", "MeanIoU", "GlobalDice"] if c in out.columns]
best = out.sort_values(rank_cols, ascending=[False] * len(rank_cols)).iloc[0]
best_threshold = float(best["EvalThreshold"])

best_info = {
    "RunTag": RunTag,
    "FeatureConfig": FeatureConfig,
    "ModelName": ModelName,
    "RunName": RunName,
    "BestThreshold": best_threshold,
}

for c in [
    "MeanDice",
    "MeanIoU",
    "MeanPrecision",
    "MeanRecall",
    "GlobalDice",
    "GlobalIoU",
    "GlobalPrecision",
    "GlobalRecall",
]:
    if c in best:
        best_info[c] = float(best[c])

best_path = out_dir / "BestThresholdSelection.json"
best_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")

src_dir = root / "Metrics" / f"Threshold_{best_threshold:.2f}"

if not src_dir.exists():
    candidates = sorted((root / "Metrics").glob(f"Threshold_{best_threshold:g}"))
    if candidates:
        src_dir = candidates[0]

src_summary = src_dir / f"{Split}MetricsSummary.csv"
src_by_sample = src_dir / f"{Split}MetricsBySample.csv"

if not src_summary.exists() or not src_by_sample.exists():
    raise FileNotFoundError(f"No encontré métricas para best threshold {best_threshold}")

active_summary = root / "Metrics" / f"{Split}MetricsSummary.csv"
active_by_sample = root / "Metrics" / f"{Split}MetricsBySample.csv"
active_eval = root / "Tables" / f"{Split}EvaluationSummary.csv"

active_summary.write_bytes(src_summary.read_bytes())
active_by_sample.write_bytes(src_by_sample.read_bytes())
active_eval.write_bytes(src_summary.read_bytes())

print(out.to_string(index=False))
print("")
print("Saved:", summary_path)
print("Saved:", best_path)
print("Promoted best threshold:", best_threshold)
PY
}

VisualizeBest() {
  local FeatureConfig="$1"
  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  BestThreshold="$(python - <<PY
from pathlib import Path
import json
path = Path("${ModelRoot}") / "Tables" / "BestThresholdSelection.json"
data = json.loads(path.read_text(encoding="utf-8"))
print(f'{float(data["BestThreshold"]):.2f}')
PY
)"

  EchoLine "VISUALIZE BEST | ${FeatureConfig} | threshold=${BestThreshold}"

  python Scripts/Step12SelectVisualizationCases.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Split "${Split}" \
    --FixedCount 9 \
    --BestCount 3 \
    --WorstCount 3 \
    --ErrorCount 3

  python Scripts/Step13VisualizePredictions.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Split "${Split}" \
    --Checkpoint "${Checkpoint}" \
    --Threshold "${BestThreshold}"
}

BuildFinalSummary() {
  EchoLine "BUILD TRANSFORMER PLUS SUMMARY"

  python - <<PY
from pathlib import Path
import pandas as pd
import json

RunTag = "${RunTag}"
ModelName = "${ModelName}"
RunName = "${RunName}"

root = Path("Outputs/Experiments") / RunTag
rows = []

for cfg in ["ConfigB", "ConfigC"]:
    model_root = root / cfg / f"{ModelName}_{RunName}"
    summary_path = model_root / "Metrics" / "TestMetricsSummary.csv"
    threshold_path = model_root / "Tables" / "BestThresholdSelection.json"
    calib_path = model_root / "Tables" / "ThresholdCalibrationSummary.csv"

    if not summary_path.exists():
        continue

    df = pd.read_csv(summary_path)
    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()

    if threshold_path.exists():
        info = json.loads(threshold_path.read_text(encoding="utf-8"))
        row["BestThreshold"] = info.get("BestThreshold")

    row["ModelDirectory"] = str(model_root)
    row["ThresholdCalibrationSummary"] = str(calib_path)
    rows.append(row)

if not rows:
    raise SystemExit("No summaries found.")

out = pd.DataFrame(rows)

cols = [
    "FeatureConfig",
    "ModelName",
    "RunName",
    "BestThreshold",
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
    "GroundTruthPixels",
    "PredictedPixels",
    "TruePositivePixels",
    "FalsePositivePixels",
    "FalseNegativePixels",
    "ModelDirectory",
    "ThresholdCalibrationSummary",
]
cols = [c for c in cols if c in out.columns]
out = out[cols].sort_values("MeanDice", ascending=False)

tables_dir = root / "Tables"
tables_dir.mkdir(parents=True, exist_ok=True)

out_path = tables_dir / "TransformerPlus_ConfigB_ConfigC_Summary.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("")
print("Saved:", out_path)
PY
}

EchoLine "TRANSFORMER PLUS EXPERIMENT START"
echo "RunTag: ${RunTag}"
echo "SourceRunTag: ${SourceRunTag}"
echo "RunName: ${RunName}"

CompileCheck
PrepareRunTag

for FeatureConfig in ConfigB ConfigC; do
  BuildAndCheckFeatures "${FeatureConfig}"
  TrainTransformerPlus "${FeatureConfig}"
  EvaluateThresholds "${FeatureConfig}"
  BuildThresholdSummaryAndPromoteBest "${FeatureConfig}"
  VisualizeBest "${FeatureConfig}"
done

BuildFinalSummary

EchoLine "TRANSFORMER PLUS EXPERIMENT COMPLETED"
echo "RunTag: ${RunTag}"
echo "Outputs: Outputs/Experiments/${RunTag}/"
