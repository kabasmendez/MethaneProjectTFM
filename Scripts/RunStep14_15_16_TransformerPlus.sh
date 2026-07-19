#!/usr/bin/env bash
set -euo pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="101840"
ComparisonTag="TransformerPlus_ConfigB_vs_ConfigC_101840"

ModelName="TransformerUNet"
RunName="Bs4Ep50_TL4_TH4_SE_D010_LR5e5_TransformerPlus"
ModelRunId="${ModelName}_${RunName}"

echo ""
echo "============================================================"
echo "COMPILE STEP14/15/16"
echo "============================================================"

python -m py_compile Scripts/Step14BuildRunHtmlReport.py
python -m py_compile Scripts/Step15CompareExperiments.py
python -m py_compile Scripts/Step16BuildComparisonHtmlReport.py

echo ""
echo "============================================================"
echo "NORMALIZE PREDICTION FIGURE INDEXES"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

run_tag = "101840"
model_run_id = "TransformerUNet_Bs4Ep50_TL4_TH4_SE_D010_LR5e5_TransformerPlus"

root = Path("Outputs/Experiments") / run_tag

for cfg in ["ConfigB", "ConfigC"]:
    model_root = root / cfg / model_run_id
    source_index = model_root / "Figures" / "PredictionCases" / "PredictionCaseIndex.csv"
    target_index = model_root / "Tables" / "PredictionFigureIndex.csv"

    if not source_index.exists():
        print("SKIP missing:", source_index)
        continue

    df = pd.read_csv(source_index)

    if "FigurePath" not in df.columns:
        raise ValueError(f"FigurePath no existe en {source_index}")

    normalized = []

    for value in df["FigurePath"].astype(str):
        p = Path(value)

        # Caso 1: ya viene relativo al model_root, como Figures/PredictionCases/x.png
        if str(value).startswith("Figures/"):
            normalized.append(str(p).replace("\\", "/"))
            continue

        # Caso 2: viene relativo al proyecto, como Outputs/Experiments/...
        try:
            rel = p.relative_to(model_root)
            normalized.append(str(rel).replace("\\", "/"))
            continue
        except Exception:
            pass

        # Caso 3: viene absoluto o relativo con prefijo de cwd
        try:
            rel = p.resolve().relative_to(model_root.resolve())
            normalized.append(str(rel).replace("\\", "/"))
            continue
        except Exception:
            pass

        # Fallback: buscar solo el nombre en PredictionCases
        candidate = model_root / "Figures" / "PredictionCases" / p.name
        if candidate.exists():
            normalized.append(str(candidate.relative_to(model_root)).replace("\\", "/"))
        else:
            normalized.append(str(value).replace("\\", "/"))

    df["FigurePath"] = normalized

    target_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_index, index=False)

    print("Wrote:", target_index)
    print("Rows:", len(df))
PY

echo ""
echo "============================================================"
echo "RUN STEP14 INDIVIDUAL HTML REPORTS"
echo "============================================================"

for FeatureConfig in ConfigB ConfigC; do
  echo ""
  echo "---- Step14 ${FeatureConfig} ----"

  python Scripts/Step14BuildRunHtmlReport.py \
    --RunTag "${RunTag}" \
    --FeatureConfig "${FeatureConfig}" \
    --ModelName "${ModelName}" \
    --RunName "${RunName}" \
    --MaxPredictionLinks 60
done

echo ""
echo "============================================================"
echo "RUN STEP15 COMPARISON"
echo "============================================================"

python Scripts/Step15CompareExperiments.py \
  --ComparisonTag "${ComparisonTag}" \
  --Items \
    "${RunTag}:ConfigB:${ModelRunId}" \
    "${RunTag}:ConfigC:${ModelRunId}"

echo ""
echo "============================================================"
echo "RUN STEP16 COMPARISON HTML REPORT"
echo "============================================================"

python Scripts/Step16BuildComparisonHtmlReport.py \
  --ComparisonTag "${ComparisonTag}" \
  --MaxVisualCases 3

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "ConfigB report:"
echo "Outputs/Experiments/${RunTag}/ConfigB/${ModelRunId}/Reports/RunReport.html"
echo ""
echo "ConfigC report:"
echo "Outputs/Experiments/${RunTag}/ConfigC/${ModelRunId}/Reports/RunReport.html"
echo ""
echo "Comparison report:"
echo "Outputs/Comparisons/${ComparisonTag}/Reports/ComparisonReport.html"
