# Official Pipeline - ConfigB Unsupervised MBMPPlus

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
