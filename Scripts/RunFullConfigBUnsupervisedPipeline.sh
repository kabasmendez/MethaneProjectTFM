#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="${1:-Exp271431}"
Epochs="${2:-30}"
BatchSize="${3:-4}"

echo "============================================================"
echo "FULL CONFIGB UNSUPERVISED PIPELINE"
echo "RunTag: ${RunTag}"
echo "Epochs: ${Epochs}"
echo "BatchSize: ${BatchSize}"
echo "============================================================"

mkdir -p "Outputs/Experiments/${RunTag}/Logs"

echo ""
echo "STEP 0 - Compile clean scripts"
python -m py_compile \
  Source/FeatureEngineering.py \
  Scripts/Step06BuildFeatureReadyNoGroundTruthFilter.py \
  Scripts/Step07BuildFeaturesClean.py \
  Scripts/Step08CheckFeatureTensorsClean.py \
  Scripts/Step09CheckDataLoadersClean.py \
  Scripts/Step10TrainSegmentationModelClean.py \
  Scripts/Step11EvaluateSegmentationModelClean.py

echo ""
echo "STEP 1 - Project audit"
python Scripts/Step00ProjectAudit.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 2 - Build dataset index"
python Scripts/Step01BuildDatasetIndex.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 3 - Inspect metadata"
python Scripts/Step02InspectMetadata.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 4 - Build FeatureReady splits without ground-truth filter"
python Scripts/Step06BuildFeatureReadyNoGroundTruthFilter.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 5 - Build ConfigB tensors with unsupervised MBMPPlus"
python Scripts/Step07BuildFeaturesClean.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 6 - Check feature tensors"
python Scripts/Step08CheckFeatureTensorsClean.py \
  --RunTag "${RunTag}"

echo ""
echo "STEP 7 - Check dataloaders"
python Scripts/Step09CheckDataLoadersClean.py \
  --RunTag "${RunTag}" \
  --BatchSize "${BatchSize}"

echo ""
echo "STEP 8 - Train and evaluate models"

run_model () {
  local ModelName="$1"
  local RunName="$2"
  local Dropout="$3"
  local LR="$4"
  local BaseChannels="$5"

  echo ""
  echo "------------------------------------------------------------"
  echo "TRAIN MODEL: ${ModelName}_${RunName}"
  echo "Dropout: ${Dropout} | LR: ${LR} | BaseChannels: ${BaseChannels}"
  echo "------------------------------------------------------------"

  python Scripts/Step10TrainSegmentationModelClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig ConfigB \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Epochs "${Epochs}" \
    --BatchSize "${BatchSize}" \
    --LearningRate "${LR}" \
    --WeightDecay 1e-5 \
    --Dropout "${Dropout}" \
    --BaseChannels "${BaseChannels}" \
    --DiceWeight 1.0 \
    --Threshold 0.5

  echo ""
  echo "EVALUATE MODEL: ${ModelName}_${RunName}"

  python Scripts/Step11EvaluateSegmentationModelClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig ConfigB \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Split Test \
    --Checkpoint BestModel.pt \
    --BatchSize 8 \
    --Threshold 0.5
}

run_model "SimpleUNet"   "Bs${BatchSize}Ep${Epochs}_D005_LR1e4_UnsupervisedMBMPPlus" "0.05" "1e-4" "32"
run_model "EnhancedUNet" "Bs${BatchSize}Ep${Epochs}_D005_LR1e4_UnsupervisedMBMPPlus" "0.05" "1e-4" "32"

echo ""
echo "STEP 9 - Optional LegacyUNet if available"

python - <<'PY'
from Source.Models.ModelFactory import CreateModel
try:
    model = CreateModel(
        ModelName="LegacyUNet",
        InputChannels=9,
        OutputChannels=1,
        ModelParameters={"BaseChannels": 32, "Dropout": 0.0},
    )
    print("LEGACY_AVAILABLE=1")
except Exception as e:
    print("LEGACY_AVAILABLE=0")
    print(e)
PY

if python - <<'PY' | grep -q "YES"
from Source.Models.ModelFactory import CreateModel
try:
    CreateModel(
        ModelName="LegacyUNet",
        InputChannels=9,
        OutputChannels=1,
        ModelParameters={"BaseChannels": 32, "Dropout": 0.0},
    )
    print("YES")
except Exception:
    print("NO")
PY
then
  run_model "LegacyUNet" "Bs${BatchSize}Ep${Epochs}_D000_LR1e4_UnsupervisedMBMPPlus" "0.00" "1e-4" "32"
else
  echo "LegacyUNet not available in ModelFactory. Skipping."
fi

echo ""
echo "STEP 10 - Build comparison summary"

python - <<PY
from pathlib import Path
import pandas as pd

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag / "ConfigB"
OutDir = Path("Outputs/Experiments") / RunTag / "Compare" / "Tables"
OutDir.mkdir(parents=True, exist_ok=True)

Rows = []

for SummaryPath in Root.glob("*/Metrics/TestMetricsSummary.csv"):
    try:
        df = pd.read_csv(SummaryPath)
        row = df.iloc[0].to_dict()

        RunDir = SummaryPath.parents[1]
        ModelRunSummaryPath = RunDir / "Tables" / "ModelRunSummary.csv"
        if ModelRunSummaryPath.exists():
            run_info = pd.read_csv(ModelRunSummaryPath).iloc[0].to_dict()
            row.update({
                "Epochs": run_info.get("Epochs"),
                "BatchSize": run_info.get("BatchSize"),
                "LearningRate": run_info.get("LearningRate"),
                "Dropout": run_info.get("Dropout"),
                "BaseChannels": run_info.get("BaseChannels"),
                "BestEpoch": run_info.get("BestEpoch"),
                "BestValidationMeanDice": run_info.get("BestValidationMeanDice"),
            })

        Rows.append(row)
    except Exception as e:
        print("Skipping", SummaryPath, e)

if not Rows:
    raise SystemExit("No TestMetricsSummary.csv found.")

Compare = pd.DataFrame(Rows)
SortColumns = [c for c in ["MeanDice", "GlobalDice"] if c in Compare.columns]
Compare = Compare.sort_values(SortColumns, ascending=False)

OutPath = OutDir / "ConfigBUnsupervisedModelComparison.csv"
Compare.to_csv(OutPath, index=False)

print("\\n=== CONFIGB UNSUPERVISED MODEL COMPARISON ===")
print(Compare.to_string(index=False))
print("\\nSaved:", OutPath)
PY

echo ""
echo "============================================================"
echo "PIPELINE COMPLETED"
echo "RunTag: ${RunTag}"
echo "Comparison: Outputs/Experiments/${RunTag}/Compare/Tables/ConfigBUnsupervisedModelComparison.csv"
echo "============================================================"
