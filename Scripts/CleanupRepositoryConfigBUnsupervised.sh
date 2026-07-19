#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"

if [[ "$MODE" != "dry-run" && "$MODE" != "apply" ]]; then
  echo "Uso:"
  echo "  bash Scripts/CleanupRepositoryConfigBUnsupervised.sh dry-run"
  echo "  bash Scripts/CleanupRepositoryConfigBUnsupervised.sh apply"
  exit 1
fi

Timestamp="$(date +%Y%m%d_%H%M%S)"
ArchiveRoot="Archive/Deprecated_${Timestamp}"

echo "============================================================"
echo "CLEANUP REPOSITORY - CONFIGB UNSUPERVISED MBMPPLUS"
echo "Mode: $MODE"
echo "ArchiveRoot: $ArchiveRoot"
echo "============================================================"

move_file () {
  local src="$1"
  local dst_dir="$2"

  if [[ ! -e "$src" ]]; then
    return 0
  fi

  echo "ARCHIVE: $src -> $dst_dir/$(basename "$src")"

  if [[ "$MODE" == "apply" ]]; then
    mkdir -p "$dst_dir"
    mv -n "$src" "$dst_dir/"
  fi
}

echo ""
echo "STEP 1 - Archive backup, patch and rewrite files"

for f in Scripts/*before* Scripts/*backup* Scripts/*failed* Scripts/*patch* Scripts/*pre_rewrite* Scripts/*pre_unsupervised* Scripts/*old*; do
  move_file "$f" "$ArchiveRoot/Scripts"
done

for f in Source/*before* Source/*backup* Source/*failed* Source/*patch* Source/*pre_* Source/*old* Source/FeatureEngineeringClean.py; do
  move_file "$f" "$ArchiveRoot/Source"
done

for f in Configs/*before* Configs/*backup* Configs/*pre_* Configs/*old*; do
  move_file "$f" "$ArchiveRoot/Configs"
done

echo ""
echo "STEP 2 - Archive deprecated scripts not used in official pipeline"

for f in \
  Scripts/Step05DiagnoseMBMPPlusValidity.py \
  Scripts/Step06ApplyFeatureReadinessFilter.py \
  Scripts/Step07BuildFeatures.py \
  Scripts/Step08CheckFeatureTensors.py \
  Scripts/Step09CheckDataLoaders.py \
  Scripts/Step10TrainSegmentationModel.py \
  Scripts/Step11EvaluateSegmentationModel.py \
  Scripts/Step11QuickPredictSegmentationModel.py \
  Scripts/RunBatchTransformerUNetEp30.sh \
  Scripts/RunFullMBMPUnsupUnetsExp271431.sh \
  Scripts/RunSimpleUNetEp30.sh \
  Scripts/RunSmokePipeline.sh \
  Scripts/RunTransformerUNetTL3Ep30.sh
do
  move_file "$f" "$ArchiveRoot/Scripts_Deprecated"
done

echo ""
echo "STEP 3 - Create official pipeline documentation"

if [[ "$MODE" == "apply" ]]; then
  mkdir -p Docs
  python3 - <<PYDOC
from pathlib import Path
doc = Path("Docs/OfficialPipeline_ConfigB_UnsupervisedMBMPPlus.md")
doc.write_text("""# Official Pipeline - ConfigB Unsupervised MBMPPlus

## Methodological decision

ConfigB uses MBMPPlus in an unsupervised way.

- MBMPPlus does not use Plume.
- MBMPPlus does not use GroundTruth.
- MBMPPlus does not use the target mask to estimate background.
- Plume is used only as segmentation target Y during training/evaluation.

## ConfigB input channels

1. B8A
2. B11
3. B12
4. NDSWIR
5. RatioB12B11
6. RatioB12B8A
7. MBMP
8. MBMPPlus
9. DualEnhancementB12B11

## Official full pipeline

Run:

    bash Scripts/RunFullConfigBUnsupervisedPipeline.sh <RunTag> <Epochs> <BatchSize>

Example:

    bash Scripts/RunFullConfigBUnsupervisedPipeline.sh Exp271431 30 4

## Official step scripts

- Step00ProjectAudit.py
- Step01BuildDatasetIndex.py
- Step02InspectMetadata.py
- Step06BuildFeatureReadyNoGroundTruthFilter.py
- Step07BuildFeaturesClean.py
- Step08CheckFeatureTensorsClean.py
- Step09CheckDataLoadersClean.py
- Step10TrainSegmentationModelClean.py
- Step11EvaluateSegmentationModelClean.py
- Step12SelectVisualizationCases.py
- Step13VisualizePredictions.py
- Step14BuildRunHtmlReport.py
- Step15CompareExperiments.py
- Step16BuildComparisonHtmlReport.py

Deprecated scripts and backups are archived under Archive/Deprecated_<timestamp>/.
""", encoding="utf-8")
print("Created", doc)
PYDOC
fi

echo ""
echo "STEP 4 - Final visible files"

echo ""
echo "Scripts:"
find Scripts -maxdepth 1 -type f | sort

echo ""
echo "Source:"
find Source -maxdepth 2 -type f | sort

echo ""
echo "Configs:"
find Configs -maxdepth 1 -type f | sort

echo ""
if [[ "$MODE" == "dry-run" ]]; then
  echo "DRY RUN completed. No files moved."
  echo "To apply:"
  echo "  bash Scripts/CleanupRepositoryConfigBUnsupervised.sh apply"
else
  echo "APPLY completed."
  echo "Archived files in: $ArchiveRoot"
  echo "Documentation: Docs/OfficialPipeline_ConfigB_UnsupervisedMBMPPlus.md"
fi
