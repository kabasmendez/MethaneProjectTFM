#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="Exp271431"
FeatureConfig="ConfigC"
Split="Test"
Checkpoint="BestModel.pt"
Threshold="0.5"

EchoLine() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

RunExperiment() {
  local ModelName="$1"
  local RunName="$2"
  local ExtraArgs="$3"

  EchoLine "TRAIN ${FeatureConfig} | ${ModelName} | ${RunName}"

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
    --Threshold "${Threshold}" \
    ${ExtraArgs}

  EchoLine "EVALUATE ${FeatureConfig} | ${ModelName} | ${RunName}"

  python Scripts/Step11EvaluateSegmentationModelClean.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Split "${Split}" \
    --Checkpoint "${Checkpoint}" \
    --BatchSize 8 \
    --Threshold "${Threshold}"

  EchoLine "SELECT VISUALIZATION CASES ${FeatureConfig} | ${ModelName} | ${RunName}"

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

  EchoLine "VISUALIZE PREDICTIONS ${FeatureConfig} | ${ModelName} | ${RunName}"

  python Scripts/Step13VisualizePredictions.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --Split "${Split}" \
    --Checkpoint "${Checkpoint}" \
    --Threshold "${Threshold}"
}

EchoLine "CONFIGC FULL EXPERIMENTS START"

# 1. SimpleUNet
RunExperiment \
  "SimpleUNet" \
  "Bs4Ep30_D005_LR1e4_UnsupervisedWindMBMPPlus" \
  ""

# 2. EnhancedUNet
RunExperiment \
  "EnhancedUNet" \
  "Bs4Ep30_D005_LR1e4_UnsupervisedWindMBMPPlus" \
  ""

# 3. TransformerUNet
RunExperiment \
  "TransformerUNet" \
  "Bs4Ep30_TL2_D005_LR1e4_UnsupervisedWindMBMPPlus" \
  "--TransformerLayers 2 --TransformerHeads 4"

EchoLine "CONFIGC FULL EXPERIMENTS COMPLETED"

echo ""
echo "Outputs:"
echo "  Outputs/Experiments/${RunTag}/${FeatureConfig}/"
