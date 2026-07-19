#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="Exp271431"
FeatureConfig="ConfigB"

echo "============================================================"
echo "FULL PIPELINE - MBMPPlus UNSUPERVISED"
echo "RunTag: ${RunTag}"
echo "FeatureConfig: ${FeatureConfig}"
echo "============================================================"

echo ""
echo "STEP 0 - Compile project"
python -m compileall Source Scripts Tests

echo ""
echo "STEP 0.1 - Check effective MBMPPlus override"
python - <<'INNER_PY'
import inspect
from Source.FeatureEngineering import ComputeMbmpPlus, ComputeMBMPPlus

src1 = inspect.getsource(ComputeMbmpPlus)
src2 = inspect.getsource(ComputeMBMPPlus)

print(src1)

bad_terms = ["ValidBackground", "GroundTruth", "Plume.sum", "RidgeModel.fit"]
for term in bad_terms:
    if term in src1 or term in src2:
        raise RuntimeError(f"Effective MBMPPlus still contains supervised term: {term}")

if "*args" not in src1:
    raise RuntimeError("ComputeMbmpPlus should accept *args to ignore old Plume argument.")

print("OK: effective MBMPPlus is unsupervised and ignores old extra arguments.")
INNER_PY

echo ""
echo "STEP 0.2 - Ensure LegacyUNet is registered"

python - <<'PY'
from pathlib import Path

models_dir = Path("Source/Models")
models_dir.mkdir(parents=True, exist_ok=True)

legacy_path = models_dir / "LegacyUNet.py"
source_unet_path = Path("Source/UnetModel.py")
factory_path = models_dir / "ModelFactory.py"

if not legacy_path.exists():
    if source_unet_path.exists():
        text = source_unet_path.read_text(encoding="utf-8")
        legacy_path.write_text(text, encoding="utf-8")
        print("Created:", legacy_path, "from", source_unet_path)
    else:
        raise FileNotFoundError("No existe Source/Models/LegacyUNet.py ni Source/UnetModel.py")

legacy_text = legacy_path.read_text(encoding="utf-8")

if "class LegacyUNet" not in legacy_text:
    legacy_text += r'''


class LegacyUNet(Unet):
    """
    Wrapper compatible con ModelFactory.

    Replica la U-Net del experimento anterior.
    """

    def __init__(
        self,
        InputChannels: int,
        OutputChannels: int = 1,
        BaseChannels: int = 32,
        KernelSize: int = 3,
        Dropout: float = 0.0,
        EncoderChannels=None,
        BottleneckChannels=None,
        **kwargs,
    ):
        if EncoderChannels is None:
            EncoderChannels = [
                BaseChannels,
                BaseChannels * 2,
                BaseChannels * 4,
                BaseChannels * 8,
            ]

        if BottleneckChannels is None:
            BottleneckChannels = BaseChannels * 16

        super().__init__(
            InputChannels=InputChannels,
            OutputChannels=OutputChannels,
            EncoderChannels=tuple(EncoderChannels),
            BottleneckChannels=int(BottleneckChannels),
            KernelSize=int(KernelSize),
            Dropout=float(Dropout),
        )
'''
    legacy_path.write_text(legacy_text, encoding="utf-8")
    print("Added LegacyUNet wrapper.")

factory = factory_path.read_text(encoding="utf-8")

if "from Source.Models.LegacyUNet import LegacyUNet" not in factory:
    insert_after = None
    for line in [
        "from Source.Models.SimpleUNet import SimpleUNet\n",
        "from Source.Models.EnhancedUNet import EnhancedUNet\n",
        "from Source.Models.TransformerUNet import TransformerUNet\n",
    ]:
        if line in factory:
            insert_after = line
            break

    if insert_after is None:
        raise RuntimeError("No encontré dónde insertar import LegacyUNet en ModelFactory.py")

    factory = factory.replace(
        insert_after,
        insert_after + "from Source.Models.LegacyUNet import LegacyUNet\n",
    )

if '"LegacyUNet": LegacyUNet,' not in factory:
    if '"SimpleUNet": SimpleUNet,' in factory:
        factory = factory.replace(
            '"SimpleUNet": SimpleUNet,',
            '"SimpleUNet": SimpleUNet,\n    "LegacyUNet": LegacyUNet,',
        )
    elif '"EnhancedUNet": EnhancedUNet,' in factory:
        factory = factory.replace(
            '"EnhancedUNet": EnhancedUNet,',
            '"EnhancedUNet": EnhancedUNet,\n    "LegacyUNet": LegacyUNet,',
        )
    else:
        raise RuntimeError("No encontré diccionario de modelos para registrar LegacyUNet.")

factory_path.write_text(factory, encoding="utf-8")
print("LegacyUNet registration checked.")
PY

python -m compileall Source Scripts Tests

echo ""
echo "STEP 0.3 - Test model creation"

python - <<'PY'
import torch
from Source.Models.ModelFactory import CreateModel

for ModelName in ["LegacyUNet", "SimpleUNet", "EnhancedUNet"]:
    model = CreateModel(
        ModelName=ModelName,
        InputChannels=9,
        OutputChannels=1,
        ModelParameters={
            "BaseChannels": 32,
            "Dropout": 0.05,
        },
    )
    x = torch.randn(2, 9, 200, 200)
    y = model(x)
    print(ModelName, y.shape)
    assert tuple(y.shape) == (2, 1, 200, 200)

print("All U-Net models OK")
PY

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
echo "STEP 4.1 - Create compatibility FeatureReady split names for Step07"

python - <<'PY'
from pathlib import Path
import shutil
import pandas as pd

RunTag = "Exp271431"
Tables = Path("Outputs/Experiments") / RunTag / "Tables"

pairs = {
    "Train": ("FeatureReadyTrain.csv", "SplitTrainFeatureReady.csv"),
    "Validation": ("FeatureReadyValidation.csv", "SplitValidationFeatureReady.csv"),
    "Test": ("FeatureReadyTest.csv", "SplitTestFeatureReady.csv"),
}

for Split, (src, dst) in pairs.items():
    src_path = Tables / src
    dst_path = Tables / dst

    if not src_path.exists():
        raise FileNotFoundError(src_path)

    shutil.copy2(src_path, dst_path)
    print(f"{Split}: {src_path.name} -> {dst_path.name}")

summary = pd.read_csv(Tables / "FeatureReadySummary.csv")
print()
print(summary.to_string(index=False))
PY

echo ""
echo "STEP 5 - Build ConfigB feature tensors"
python Scripts/Step07BuildFeatures.py \
  --RunTag "${RunTag}" \
  --FeatureConfigs "${FeatureConfig}" \
  --UseFeatureReadySplits

echo ""
echo "STEP 6 - Check feature tensors"
if python Scripts/Step08CheckFeatureTensors.py -h 2>&1 | grep -q -- "--FeatureConfigs"; then
    python Scripts/Step08CheckFeatureTensors.py \
      --RunTag "${RunTag}" \
      --FeatureConfigs "${FeatureConfig}"
elif python Scripts/Step08CheckFeatureTensors.py -h 2>&1 | grep -q -- "--Configs"; then
    python Scripts/Step08CheckFeatureTensors.py \
      --RunTag "${RunTag}" \
      --Configs "${FeatureConfig}"
else
    python Scripts/Step08CheckFeatureTensors.py \
      --RunTag "${RunTag}"
fi

echo ""
echo "STEP 7 - Check dataloaders"
if python Scripts/Step09CheckDataLoaders.py -h 2>&1 | grep -q -- "--FeatureConfigs"; then
    python Scripts/Step09CheckDataLoaders.py \
      --RunTag "${RunTag}" \
      --FeatureConfigs "${FeatureConfig}"
elif python Scripts/Step09CheckDataLoaders.py -h 2>&1 | grep -q -- "--Configs"; then
    python Scripts/Step09CheckDataLoaders.py \
      --RunTag "${RunTag}" \
      --Configs "${FeatureConfig}"
else
    python Scripts/Step09CheckDataLoaders.py \
      --RunTag "${RunTag}"
fi

echo ""
echo "STEP 7.1 - Manual tensor verification"

python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

RunTag = "Exp271431"
Root = Path("Outputs/Experiments") / RunTag
FeatureDir = Root / "ConfigB" / "Features"

print("\n=== FeatureReady summary ===")
print(pd.read_csv(Root / "Tables" / "FeatureReadySummary.csv").to_string(index=False))

print("\n=== Tensor check ===")
for Split in ["Train", "Validation", "Test"]:
    X = np.load(FeatureDir / f"{Split}Features.npy", mmap_mode="r")
    Y = np.load(FeatureDir / f"{Split}Masks.npy", mmap_mode="r")

    print(f"\n{Split}")
    print("X:", X.shape, X.dtype, "min:", float(np.nanmin(X)), "max:", float(np.nanmax(X)))
    print("Y:", Y.shape, Y.dtype, "positive:", int(Y.sum()))
    print("Finite:", bool(np.isfinite(X).all()))

    expected = {
        "Train": 2463,
        "Validation": 528,
        "Test": 528,
    }[Split]

    assert X.shape == (expected, 9, 200, 200), f"{Split} X shape inesperada: {X.shape}"
    assert Y.shape == (expected, 1, 200, 200), f"{Split} Y shape inesperada: {Y.shape}"
    assert np.isfinite(X).all(), f"{Split} contiene NaN/Inf"

print("\nFULL FEATURE BUILD VERIFIED OK.")
PY

echo ""
echo "STEP 8 - Train and evaluate U-Net models"

run_unet () {
    local ModelName="$1"
    local RunName="$2"
    local Dropout="$3"
    local LearningRate="$4"

    echo ""
    echo "============================================================"
    echo "START MODEL"
    echo "ModelName: ${ModelName}"
    echo "RunName: ${RunName}"
    echo "============================================================"

    python Scripts/Step10TrainSegmentationModel.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --Epochs 30 \
      --BatchSize 4 \
      --BaseChannels 32 \
      --Dropout "${Dropout}" \
      --LearningRate "${LearningRate}" \
      --WeightDecay 1e-5 \
      --Threshold 0.5

    python Scripts/Step11EvaluateSegmentationModel.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --Split Test \
      --Checkpoint BestModel.pt \
      --BatchSize 8 \
      --Threshold 0.5

    python Scripts/Step12SelectVisualizationCases.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --FixedCount 12 \
      --BestCount 6 \
      --WorstCount 6 \
      --ErrorCount 6

    python Scripts/Step13VisualizePredictions.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --Split Test \
      --Checkpoint BestModel.pt \
      --Threshold 0.5 \
      --CaseGroups FixedComparisonCases BestPredictions WorstPredictions \
      --MaxPerGroup 12

    python Scripts/Step14BuildRunHtmlReport.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --MaxPredictionLinks 30

    echo ""
    echo "COMPLETED: ${ModelName}_${RunName}"
    echo "Dashboard:"
    echo "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Reports/RunReport.html"
}

run_unet "LegacyUNet"   "Bs4Ep30_D000_LR1e4" "0.00" "1e-4"
run_unet "SimpleUNet"   "Bs4Ep30_D005_LR1e4" "0.05" "1e-4"
run_unet "EnhancedUNet" "Bs4Ep30_D005_LR1e4" "0.05" "1e-4"

echo ""
echo "STEP 9 - Final summary"

python - <<'PY'
from pathlib import Path
import pandas as pd

RunTag = "Exp271431"
Config = "ConfigB"

Runs = [
    "LegacyUNet_Bs4Ep30_D000_LR1e4",
    "SimpleUNet_Bs4Ep30_D005_LR1e4",
    "EnhancedUNet_Bs4Ep30_D005_LR1e4",
]

Root = Path("Outputs/Experiments") / RunTag / Config
Rows = []

for Run in Runs:
    ModelRoot = Root / Run

    TestPath = ModelRoot / "Metrics" / "TestMetricsSummary.csv"
    BestPath = ModelRoot / "Metrics" / "BestEpochSummary.csv"
    SummaryPath = ModelRoot / "Tables" / "ModelRunSummary.csv"

    if not TestPath.exists():
        Rows.append({"ModelRunId": Run, "Status": "Missing TestMetricsSummary"})
        continue

    T = pd.read_csv(TestPath).iloc[0].to_dict()
    B = pd.read_csv(BestPath).iloc[0].to_dict() if BestPath.exists() else {}
    S = pd.read_csv(SummaryPath).iloc[0].to_dict() if SummaryPath.exists() else {}

    Rows.append({
        "ModelRunId": Run,
        "ModelName": S.get("ModelName", ""),
        "Epochs": S.get("Epochs", ""),
        "BatchSize": S.get("BatchSize", ""),
        "LearningRate": S.get("LearningRate", ""),
        "Dropout": S.get("Dropout", ""),
        "BestEpoch": B.get("BestEpoch", ""),
        "BestValidationMeanDice": B.get("BestValidationMeanDice", ""),
        "BestValidationGlobalDice": B.get("BestValidationGlobalDice", ""),
        "Test_MeanDice": T.get("MeanDice", ""),
        "Test_MeanIoU": T.get("MeanIoU", ""),
        "Test_GlobalDice": T.get("GlobalDice", ""),
        "Test_GlobalIoU": T.get("GlobalIoU", ""),
        "Test_MeanPrecision": T.get("MeanPrecision", ""),
        "Test_MeanRecall": T.get("MeanRecall", ""),
    })

df = pd.DataFrame(Rows)

OutDir = Path("Outputs/Experiments") / RunTag / "Compare" / "Tables"
OutDir.mkdir(parents=True, exist_ok=True)
OutPath = OutDir / "MBMPUnsupUnetSummary.csv"
df.to_csv(OutPath, index=False)

print("\n=== MBMP UNSUPERVISED U-NET SUMMARY ===")
if "Test_MeanDice" in df.columns:
    print(df.sort_values("Test_MeanDice", ascending=False).to_string(index=False))
else:
    print(df.to_string(index=False))

print("\nSaved:", OutPath)
PY

echo ""
echo "============================================================"
echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "RunTag: ${RunTag}"
echo "Description: ConfigB with unsupervised MBMPPlus and no ground-truth feature filter"
echo "============================================================"
