#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

Today="$(date +%Y%m%d)"
RunTag="${1:-Exp${Today}_ConfigBConfigC_Full}"
SourceRunTag="${2:-Exp271431}"

Split="Test"
Checkpoint="BestModel.pt"
DefaultTrainThreshold="0.50"
Thresholds=("0.30" "0.40" "0.50" "0.60" "0.70")

LogDir="Outputs/Experiments/${RunTag}/Logs"
mkdir -p "${LogDir}"

EchoLine() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

RequireFile() {
  local Path="$1"
  if [[ ! -f "$Path" ]]; then
    echo "ERROR: falta archivo requerido: $Path"
    exit 1
  fi
}

CompileScripts() {
  EchoLine "COMPILE CHECK"

  python -m py_compile Scripts/Step07BuildFeaturesClean.py
  python -m py_compile Scripts/Step08CheckFeatureTensorsClean.py
  python -m py_compile Scripts/Step10TrainSegmentationModelClean.py
  python -m py_compile Scripts/Step11EvaluateSegmentationModelClean.py
  python -m py_compile Scripts/Step12SelectVisualizationCases.py
  python -m py_compile Scripts/Step13VisualizePredictions.py

  echo "Compile OK."
}

PrepareRunTables() {
  EchoLine "PREPARE NEW RUN TAG: ${RunTag}"

  mkdir -p "Outputs/Experiments/${RunTag}/Tables"

  for File in \
    SplitTrainFeatureReady.csv \
    SplitValidationFeatureReady.csv \
    SplitTestFeatureReady.csv
  do
    RequireFile "Outputs/Experiments/${SourceRunTag}/Tables/${File}"
    cp "Outputs/Experiments/${SourceRunTag}/Tables/${File}" \
       "Outputs/Experiments/${RunTag}/Tables/${File}"
    echo "Copied: ${File}"
  done

  cat > "Outputs/Experiments/${RunTag}/RunMetadata.txt" <<EOF
RunTag=${RunTag}
SourceRunTag=${SourceRunTag}
CreatedAt=$(date -Iseconds)
Purpose=Full dated experiment for ConfigB and ConfigC with all models.
Models=SimpleUNet, EnhancedUNet, TransformerUNet
FeatureConfigs=ConfigB, ConfigC
Thresholds=${Thresholds[*]}
EOF

  echo "Prepared tables and metadata for ${RunTag}"
}

BuildAndCheckFeatures() {
  local FeatureConfig="$1"

  EchoLine "BUILD FEATURES | ${FeatureConfig}"

  python Scripts/Step07BuildFeaturesClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --UseFeatureReadySplits \
    --ClipValue 8.0

  EchoLine "CHECK FEATURE TENSORS | ${FeatureConfig}"

  python Scripts/Step08CheckFeatureTensorsClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}"
}

TrainModel() {
  local FeatureConfig="$1"
  local ModelName="$2"
  local RunName="$3"
  local ExtraArgs="$4"

  EchoLine "TRAIN | ${FeatureConfig} | ${ModelName} | ${RunName}"

  python Scripts/Step10TrainSegmentationModelClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Epochs 30 \
    --BatchSize 4 \
    --LearningRate 1e-4 \
    --WeightDecay 1e-5 \
    --Dropout 0.05 \
    --BaseChannels 32 \
    --DiceWeight 1.0 \
    --Threshold "${DefaultTrainThreshold}" \
    ${ExtraArgs}
}

EvaluateThresholds() {
  local FeatureConfig="$1"
  local ModelName="$2"
  local RunName="$3"

  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  EchoLine "EVALUATE THRESHOLDS | ${FeatureConfig} | ${ModelName} | ${RunName}"

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
  local ModelName="$2"
  local RunName="$3"

  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  EchoLine "BUILD THRESHOLD SUMMARY AND PROMOTE BEST | ${FeatureConfig} | ${ModelName} | ${RunName}"

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

# Best threshold por MeanDice, desempate por MeanIoU y GlobalDice.
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
for c in ["MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall", "GlobalDice", "GlobalIoU", "GlobalPrecision", "GlobalRecall"]:
    if c in best:
        best_info[c] = float(best[c])

best_path = out_dir / "BestThresholdSelection.json"
best_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")

# Promover el threshold seleccionado como resultado activo.
src_summary = root / "Metrics" / f"Threshold_{best_threshold:.2f}" / f"{Split}MetricsSummary.csv"
src_by_sample = root / "Metrics" / f"Threshold_{best_threshold:.2f}" / f"{Split}MetricsBySample.csv"

if not src_summary.exists():
    # Fallback por si float format cambia, buscar carpeta compatible.
    candidates = sorted((root / "Metrics").glob(f"Threshold_{best_threshold:g}"))
    if candidates:
        src_summary = candidates[0] / f"{Split}MetricsSummary.csv"
        src_by_sample = candidates[0] / f"{Split}MetricsBySample.csv"

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
print("Active summary:", active_summary)
print("Active by sample:", active_by_sample)
print("Active evaluation:", active_eval)
PY
}

VisualizeBestThreshold() {
  local FeatureConfig="$1"
  local ModelName="$2"
  local RunName="$3"

  local ModelRoot="Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}"

  BestThreshold="$(python - <<PY
from pathlib import Path
import json
path = Path("${ModelRoot}") / "Tables" / "BestThresholdSelection.json"
data = json.loads(path.read_text(encoding="utf-8"))
print(f'{float(data["BestThreshold"]):.2f}')
PY
)"

  EchoLine "SELECT CASES AND VISUALIZE | ${FeatureConfig} | ${ModelName} | ${RunName} | threshold=${BestThreshold}"

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

RunOneModel() {
  local FeatureConfig="$1"
  local ModelName="$2"
  local RunName="$3"
  local ExtraArgs="$4"

  TrainModel "${FeatureConfig}" "${ModelName}" "${RunName}" "${ExtraArgs}"
  EvaluateThresholds "${FeatureConfig}" "${ModelName}" "${RunName}"
  BuildThresholdSummaryAndPromoteBest "${FeatureConfig}" "${ModelName}" "${RunName}"
  VisualizeBestThreshold "${FeatureConfig}" "${ModelName}" "${RunName}"
}

BuildGlobalSummary() {
  EchoLine "BUILD GLOBAL SUMMARY"

  python - <<PY
from pathlib import Path
import pandas as pd
import json

RunTag = "${RunTag}"
root = Path("Outputs/Experiments") / RunTag
rows = []

for cfg in ["ConfigB", "ConfigC"]:
    cfg_root = root / cfg
    if not cfg_root.exists():
        continue

    for model_dir in sorted(cfg_root.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_path = model_dir / "Metrics" / "TestMetricsSummary.csv"
        threshold_path = model_dir / "Tables" / "BestThresholdSelection.json"
        calib_path = model_dir / "Tables" / "ThresholdCalibrationSummary.csv"

        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if len(df) == 0:
            continue

        row = df.iloc[0].to_dict()
        row["ModelDirectory"] = str(model_dir)
        row["ThresholdCalibrationSummary"] = str(calib_path) if calib_path.exists() else ""

        if threshold_path.exists():
            data = json.loads(threshold_path.read_text(encoding="utf-8"))
            row["BestThreshold"] = data.get("BestThreshold")

        rows.append(row)

if not rows:
    raise SystemExit("No active TestMetricsSummary.csv found.")

out = pd.DataFrame(rows)

cols = [
    "FeatureConfig",
    "ModelName",
    "RunName",
    "Split",
    "Checkpoint",
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
out = out[cols].sort_values(["FeatureConfig", "ModelName"])

tables_dir = root / "Tables"
tables_dir.mkdir(parents=True, exist_ok=True)

out_path = tables_dir / "FullExperiment_TestSummary_AllModels.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print("")
print("Saved:", out_path)

# Tabla pivote compacta ConfigB vs ConfigC.
compact_cols = [c for c in ["FeatureConfig", "ModelName", "BestThreshold", "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall", "GlobalDice", "GlobalIoU"] if c in out.columns]
compact = out[compact_cols].copy()
compact_path = tables_dir / "FullExperiment_CompactComparison.csv"
compact.to_csv(compact_path, index=False)
print("Saved:", compact_path)
PY
}

EchoLine "DATED FULL EXPERIMENT START"
echo "RunTag: ${RunTag}"
echo "SourceRunTag for splits: ${SourceRunTag}"
echo "LogDir: ${LogDir}"

CompileScripts
PrepareRunTables

for FeatureConfig in ConfigB ConfigC; do
  BuildAndCheckFeatures "${FeatureConfig}"

  RunOneModel \
    "${FeatureConfig}" \
    "SimpleUNet" \
    "Bs4Ep30_D005_LR1e4_DatedFull" \
    ""

  RunOneModel \
    "${FeatureConfig}" \
    "EnhancedUNet" \
    "Bs4Ep30_D005_LR1e4_DatedFull" \
    ""

  RunOneModel \
    "${FeatureConfig}" \
    "TransformerUNet" \
    "Bs4Ep30_TL2_D005_LR1e4_DatedFull" \
    "--TransformerLayers 2 --TransformerHeads 4"
done

BuildGlobalSummary

EchoLine "DATED FULL EXPERIMENT COMPLETED"
echo "RunTag: ${RunTag}"
echo "Outputs: Outputs/Experiments/${RunTag}/"
