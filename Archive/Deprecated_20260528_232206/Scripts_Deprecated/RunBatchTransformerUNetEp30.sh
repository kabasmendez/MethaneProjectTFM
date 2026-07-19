#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="Exp261944"
FeatureConfig="ConfigB"
ModelName="TransformerUNet"

run_experiment () {
    local RunName="$1"
    local Dropout="$2"
    local LearningRate="$3"

    echo ""
    echo "============================================================"
    echo "START: ${RunName}"
    echo "Dropout: ${Dropout}"
    echo "LearningRate: ${LearningRate}"
    echo "============================================================"

    python Scripts/Step10TrainSegmentationModel.py \
      --RunTag "${RunTag}" \
      --FeatureConfig "${FeatureConfig}" \
      --ModelName "${ModelName}" \
      --RunName "${RunName}" \
      --Epochs 30 \
      --BatchSize 4 \
      --BaseChannels 32 \
      --UseSqueezeExcitation \
      --Dropout "${Dropout}" \
      --TransformerHeads 8 \
      --TransformerLayers 2 \
      --TransformerMlpRatio 4.0 \
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
    echo "COMPLETED: ${RunName}"
    echo "Dashboard:"
    echo "Outputs/Experiments/${RunTag}/${FeatureConfig}/${ModelName}_${RunName}/Reports/RunReport.html"
}

run_experiment "Bs4Ep30_TL2_D005_LR1e4" "0.05" "1e-4"
run_experiment "Bs4Ep30_TL2_D010_LR7e5" "0.10" "7e-5"

echo ""
echo "============================================================"
echo "ALL EP30 EXPERIMENTS COMPLETED"
echo "============================================================"
