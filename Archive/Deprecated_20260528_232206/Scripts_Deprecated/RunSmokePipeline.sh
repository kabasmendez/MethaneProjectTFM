#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="${1:-$(date +"Exp%d%H%M")}"

echo "============================================================"
echo "MethaneProjectTFM smoke pipeline"
echo "RunTag: ${RunTag}"
echo "Root: $(pwd)"
echo "============================================================"

echo ""
echo ">>> 0. Verificando código y tests"
python -m compileall Source Scripts Tests
python -m pytest Tests -q

echo ""
echo ">>> 1. Step00ProjectAudit"
python Scripts/Step00ProjectAudit.py --RunTag "${RunTag}"

echo ""
echo ">>> 2. Step01BuildDatasetIndex"
python Scripts/Step01BuildDatasetIndex.py --RunTag "${RunTag}"

echo ""
echo ">>> 3. Verificando splits"
python - <<PY
from pathlib import Path
import pandas as pd

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag

RequiredTables = [
    "DatasetIndex.csv",
    "DatasetFiltered.csv",
    "FilterSummary.csv",
    "SplitTrain.csv",
    "SplitValidation.csv",
    "SplitTest.csv",
    "SplitSummary.csv",
]

for Name in RequiredTables:
    PathItem = Root / "Tables" / Name
    if not PathItem.exists():
        raise FileNotFoundError(PathItem)
    Table = pd.read_csv(PathItem)
    print(Name, Table.shape)

SplitSummary = pd.read_csv(Root / "Tables" / "SplitSummary.csv")
print("\\nSplitSummary:")
print(SplitSummary.to_string(index=False))
PY

echo ""
echo ">>> 4. Step02InspectMetadata"
python Scripts/Step02InspectMetadata.py --RunTag "${RunTag}"

echo ""
echo ">>> 5. Verificando metadatos contextuales"
python - <<PY
from pathlib import Path
import pandas as pd

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag
PathItem = Root / "Tables" / "ContextMetadataCheck.csv"

if not PathItem.exists():
    raise FileNotFoundError(PathItem)

Table = pd.read_csv(PathItem)
print(Table.to_string(index=False))
PY

echo ""
echo ">>> 6. Step03VisualizeSamples"
python Scripts/Step03VisualizeSamples.py --RunTag "${RunTag}" --MaxSamples 2

echo ""
echo ">>> 7. Verificando visualización de muestras"
python - <<PY
from pathlib import Path
import pandas as pd

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag

TablePath = Root / "Tables" / "VisualizedSamples.csv"
if not TablePath.exists():
    raise FileNotFoundError(TablePath)

Table = pd.read_csv(TablePath)
print(Table[[
    "SampleId",
    "TargetShape",
    "ReferenceShape",
    "PlumeShape",
    "CH4Shape",
    "PlumePixels",
    "IsValid"
]].to_string(index=False))

RequiredFigures = list((Root / "Figures").glob("SampleOverview_*.png"))
Grid = Root / "Figures" / "SampleGrid.png"

if not Grid.exists():
    raise FileNotFoundError(Grid)

if len(RequiredFigures) == 0:
    raise FileNotFoundError("No se generaron SampleOverview_*.png")

print("\\nSampleGrid:", Grid)
print("SampleOverview count:", len(RequiredFigures))
PY

echo ""
echo ">>> 8. Step04PreviewFeatures"
python Scripts/Step04PreviewFeatures.py --RunTag "${RunTag}" --MaxSamples 1

echo ""
echo ">>> 9. Verificando preview de features"
python - <<PY
from pathlib import Path
import pandas as pd
import json

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag

Expected = {
    "ConfigA": 7,
    "ConfigB": 9,
}

for Config, ExpectedCount in Expected.items():
    TablePath = Root / Config / "Tables" / "FeaturePreviewStats.csv"
    if not TablePath.exists():
        raise FileNotFoundError(TablePath)

    Table = pd.read_csv(TablePath)
    Features = Table["Feature"].drop_duplicates().tolist()

    print("\\n===", Config, "===")
    print("\\n".join(Features))

    if len(Features) != ExpectedCount:
        raise AssertionError(f"{Config} esperaba {ExpectedCount} features y obtuvo {len(Features)}")

AuditPath = Root / "Audit" / "FeaturePreviewAudit.json"
if not AuditPath.exists():
    raise FileNotFoundError(AuditPath)

Audit = json.loads(AuditPath.read_text(encoding="utf-8"))
print("\\nMBMPPlusMethodology:")
print(json.dumps(Audit["Details"]["MBMPPlusMethodology"], indent=2, ensure_ascii=False))
PY

echo ""
echo ">>> 10. Step05BuildFeatures modo seguro"
python Scripts/Step05BuildFeatures.py \
  --RunTag "${RunTag}" \
  --FeatureConfigs ConfigA ConfigB \
  --MaxSamplesPerSplit 3

echo ""
echo ">>> 11. Verificando tensores de features modo seguro"
python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd
import json

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag

Expected = {
    "Train": 3,
    "Validation": 3,
    "Test": 3,
}

for Config, Channels in [("ConfigA", 7), ("ConfigB", 9)]:
    print("\\n===", Config, "===")
    FeatureDir = Root / Config / "Features"

    for Split, N in Expected.items():
        XPath = FeatureDir / f"{Split}Features.npy"
        YPath = FeatureDir / f"{Split}Masks.npy"

        if not XPath.exists():
            raise FileNotFoundError(XPath)
        if not YPath.exists():
            raise FileNotFoundError(YPath)

        X = np.load(XPath, mmap_mode="r")
        Y = np.load(YPath, mmap_mode="r")

        print(Split)
        print("  X:", X.shape, X.dtype, "min:", float(X.min()), "max:", float(X.max()))
        print("  Y:", Y.shape, Y.dtype, "positive pixels:", int(Y.sum()))
        print("  finite X:", bool(np.isfinite(X[:]).all()))

        assert X.shape == (N, Channels, 200, 200)
        assert Y.shape == (N, 1, 200, 200)
        assert X.dtype == np.float32
        assert Y.dtype == np.uint8
        assert np.isfinite(X[:]).all()

    SummaryPath = Root / Config / "Tables" / "FeatureBuildSummary.csv"
    StatsPath = Root / Config / "Tables" / "FeatureNormalizationStats.csv"

    if not SummaryPath.exists():
        raise FileNotFoundError(SummaryPath)
    if not StatsPath.exists():
        raise FileNotFoundError(StatsPath)

    Summary = pd.read_csv(SummaryPath)
    Stats = pd.read_csv(StatsPath)

    print("\\nFeatures:")
    print("\\n".join(Stats["Feature"].tolist()))

    print("\\nSummary rows:", len(Summary))
    print("Any non-finite:", Summary["AnyNonFiniteAfterNormalization"].any())

    if bool(Summary["AnyNonFiniteAfterNormalization"].any()):
        raise AssertionError(f"{Config} tiene valores no finitos después de normalizar.")

    if Config == "ConfigB":
        AuditPath = Root / Config / "Audit" / "FeatureBuildAudit.json"
        Audit = json.loads(AuditPath.read_text(encoding="utf-8"))
        print("\\nMBMPPlus audit:")
        print(json.dumps(Audit["Details"]["MBMPPlusMethodology"], indent=2, ensure_ascii=False))

print("\\nSmoke tensor verification OK.")
PY

echo ""
echo ">>> 12. Resumen final"
python - <<PY
from pathlib import Path

RunTag = "${RunTag}"
Root = Path("Outputs/Experiments") / RunTag

print("RunTag:", RunTag)
print("Experiment root:", Root)
print("Files:", len([p for p in Root.rglob("*") if p.is_file()]))

for Subdir in ["Tables", "Figures", "Audit", "Logs", "ConfigA", "ConfigB", "ConfigC"]:
    PathItem = Root / Subdir
    print(Subdir, "exists:", PathItem.exists())

print("\\nKey outputs:")
for Pattern in [
    "Tables/SplitSummary.csv",
    "Tables/ContextMetadataCheck.csv",
    "Figures/SampleGrid.png",
    "ConfigA/Tables/FeaturePreviewStats.csv",
    "ConfigB/Tables/FeaturePreviewStats.csv",
    "ConfigA/Features/TrainFeatures.npy",
    "ConfigB/Features/TrainFeatures.npy",
]:
    PathItem = Root / Pattern
    print(PathItem, "OK" if PathItem.exists() else "MISSING")
PY

echo ""
echo "============================================================"
echo "SMOKE PIPELINE COMPLETED SUCCESSFULLY"
echo "RunTag: ${RunTag}"
echo "Outputs: Outputs/Experiments/${RunTag}"
echo "============================================================"
