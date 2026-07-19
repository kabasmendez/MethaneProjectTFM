#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="Exp261944"
FeatureConfig="ConfigB"
ModelName="SimpleUNet"
RunName="Bs4Ep30_D005_LR1e4"

echo "============================================================"
echo "SIMPLE UNET BASELINE"
echo "RunTag: ${RunTag}"
echo "FeatureConfig: ${FeatureConfig}"
echo "ModelName: ${ModelName}"
echo "RunName: ${RunName}"
echo "Epochs: 30"
echo "BatchSize: 4"
echo "BaseChannels: 32"
echo "Dropout: 0.05"
echo "LearningRate: 1e-4"
echo "============================================================"

echo ""
echo "STEP 1/5 - Training"

python Scripts/Step10TrainSegmentationModel.py \
  --RunTag "${RunTag}" \
  --FeatureConfig "${FeatureConfig}" \
  --ModelName "${ModelName}" \
  --RunName "${RunName}" \
  --Epochs 30 \
  --BatchSize 4 \
  --BaseChannels 32 \
  --Dropout 0.05 \
  --LearningRate 1e-4 \
  --WeightDecay 1e-5 \
  --Threshold 0.5

echo ""
echo "STEP 2/5 - Formal Test evaluation"

python Scripts/Step11EvaluateSegmentationModel.py \
  --RunTag "${RunTag}" \
  --FeatureConfig "${FeatureConfig}" \
  --ModelName "${ModelName}" \
  --RunName "${RunName}" \
  --Split Test \
  --Checkpoint BestModel.pt \
  --BatchSize 8 \
  --Threshold 0.5

echo ""
echo "STEP 3/5 - Select visualization cases"

python Scripts/Step12SelectVisualizationCases.py \
  --RunTag "${RunTag}" \
  --FeatureConfig "${FeatureConfig}" \
  --ModelName "${ModelName}" \
  --RunName "${RunName}" \
  --FixedCount 12 \
  --BestCount 6 \
  --WorstCount 6 \
  --ErrorCount 6

echo ""
echo "STEP 4/5 - Visualize predictions"

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

echo ""
echo "STEP 5/5 - Build HTML dashboard"

python Scripts/Step14BuildRunHtmlReport.py \
  --RunTag "${RunTag}" \
  --FeatureConfig "${FeatureConfig}" \
  --ModelName "${ModelName}" \
  --RunName "${RunName}" \
  --MaxPredictionLinks 30

echo ""
echo "============================================================"
echo "SIMPLE UNET BASELINE COMPLETED"
echo "ModelRunId: ${ModelName}_${RunName}"
echo "Dashboard:"
echo "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Reports/RunReport.html"
echo "URL:"
echo "http://localhost:8010/Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Reports/RunReport.html?v=simpleunet"
echo "============================================================"
