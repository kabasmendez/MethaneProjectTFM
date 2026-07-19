#!/usr/bin/env bash
set -u -o pipefail

cd /data/users/kabasmen/MethaneProjectTFM

RunTag="${1:-Exp271431}"
Timestamp="$(date +%Y%m%d_%H%M%S)"
AuditDir="Outputs/Experiments/${RunTag}/Audit/PreNewConfigC_${Timestamp}"

mkdir -p "${AuditDir}"

Report="${AuditDir}/AuditReport.md"
Log="${AuditDir}/AuditConsole.log"

exec > >(tee -a "${Log}") 2>&1

echo "# Pre-new ConfigC experiment audit" > "${Report}"
echo "" >> "${Report}"
echo "- RunTag: ${RunTag}" >> "${Report}"
echo "- Timestamp: ${Timestamp}" >> "${Report}"
echo "- Project: /data/users/kabasmen/MethaneProjectTFM" >> "${Report}"
echo "" >> "${Report}"

Section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
  echo "" >> "${Report}"
  echo "## $1" >> "${Report}"
  echo "" >> "${Report}"
}

AddReport() {
  echo "$1" >> "${Report}"
}

RunCheck() {
  local Name="$1"
  shift

  echo ""
  echo ">>> ${Name}"
  echo '```text' >> "${Report}"
  echo "\$ $*" >> "${Report}"

  "$@"
  local Status=$?

  echo "Exit status: ${Status}"
  echo "Exit status: ${Status}" >> "${Report}"
  echo '```' >> "${Report}"
  echo "" >> "${Report}"

  return 0
}

Section "1. Repository structure"

RequiredFiles=(
  "Configs/ConfigB.yaml"
  "Configs/ConfigC.yaml"
  "Configs/ProjectConfig.yaml"
  "Scripts/Step07BuildFeaturesClean.py"
  "Scripts/Step08CheckFeatureTensorsClean.py"
  "Scripts/Step10TrainSegmentationModelClean.py"
  "Scripts/Step11EvaluateSegmentationModelClean.py"
  "Scripts/Step12SelectVisualizationCases.py"
  "Scripts/Step13VisualizePredictions.py"
  "Source/FeatureEngineering.py"
  "Source/Models/ModelFactory.py"
)

MissingCount=0

for f in "${RequiredFiles[@]}"; do
  if [[ -f "$f" ]]; then
    echo "OK: $f"
    AddReport "- OK: \`$f\`"
  else
    echo "MISSING: $f"
    AddReport "- MISSING: \`$f\`"
    MissingCount=$((MissingCount + 1))
  fi
done

echo ""
echo "Missing required files: ${MissingCount}"
AddReport ""
AddReport "- Missing required files: ${MissingCount}"

Section "2. Python syntax compile"

CompileFiles=(
  "Scripts/Step07BuildFeaturesClean.py"
  "Scripts/Step08CheckFeatureTensorsClean.py"
  "Scripts/Step10TrainSegmentationModelClean.py"
  "Scripts/Step11EvaluateSegmentationModelClean.py"
  "Scripts/Step12SelectVisualizationCases.py"
  "Scripts/Step13VisualizePredictions.py"
  "Source/FeatureEngineering.py"
  "Source/Models/ModelFactory.py"
)

CompileFailures=0

for f in "${CompileFiles[@]}"; do
  if [[ -f "$f" ]]; then
    echo "Compiling $f"
    python -m py_compile "$f"
    Status=$?
    if [[ "$Status" -eq 0 ]]; then
      AddReport "- Compile OK: \`$f\`"
    else
      AddReport "- Compile FAIL: \`$f\`"
      CompileFailures=$((CompileFailures + 1))
    fi
  else
    AddReport "- Compile SKIP missing: \`$f\`"
  fi
done

AddReport ""
AddReport "- Compile failures: ${CompileFailures}"

Section "3. Search for risky hardcodes"

{
  echo "### grep results"
  echo '```text'
} >> "${Report}"

grep -RIn \
  "preparado para ConfigB\|FeatureConfig.*ConfigB\|InputChannels=9\|InputChannels.*9\|ConfigDict.*InputChannels.*9" \
  Scripts/ Source/ Configs/ \
  --exclude-dir="__pycache__" \
  --exclude="*.csv" \
  --exclude="*.json" \
  --exclude="*.npy" \
  2>/dev/null | tee "${AuditDir}/RiskyHardcodes.txt"

GrepStatus=${PIPESTATUS[0]}

{
  echo '```'
  echo ""
  echo "- grep exit status: ${GrepStatus} ; status 1 only means no matches."
} >> "${Report}"

Section "4. Config and feature tensor audit"

python - <<'PY' | tee "${AuditDir}/ConfigAndTensorAudit.txt"
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

run_tag = "Exp271431"
root = Path("/data/users/kabasmen/MethaneProjectTFM")
run_root = root / "Outputs" / "Experiments" / run_tag

configs = ["ConfigB", "ConfigC"]
splits = ["Train", "Validation", "Test"]

rows = []
errors = []

for cfg in configs:
    cfg_path = root / "Configs" / f"{cfg}.yaml"

    if not cfg_path.exists():
        errors.append({"Scope": "Config", "Config": cfg, "Error": f"Missing {cfg_path}"})
        continue

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    features = data.get("Features", [])
    input_channels = data.get("InputChannels", None)

    print(f"\nCONFIG {cfg}")
    print(f"  YAML InputChannels: {input_channels}")
    print(f"  YAML Features: {len(features)}")
    print(f"  Features: {features}")

    if input_channels is not None and int(input_channels) != len(features):
        errors.append({
            "Scope": "Config",
            "Config": cfg,
            "Error": f"InputChannels {input_channels} != len(Features) {len(features)}"
        })

    feature_dir = run_root / cfg / "Features"

    for split in splits:
        x_path = feature_dir / f"{split}Features.npy"
        y_path = feature_dir / f"{split}Masks.npy"

        row = {
            "FeatureConfig": cfg,
            "Split": split,
            "XPath": str(x_path),
            "YPath": str(y_path),
            "XExists": x_path.exists(),
            "YExists": y_path.exists(),
        }

        if not x_path.exists() or not y_path.exists():
            rows.append(row)
            continue

        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        row.update({
            "XShape": str(tuple(x.shape)),
            "YShape": str(tuple(y.shape)),
            "XSamples": int(x.shape[0]) if x.ndim >= 1 else None,
            "XChannels": int(x.shape[1]) if x.ndim >= 2 else None,
            "XHeight": int(x.shape[2]) if x.ndim >= 3 else None,
            "XWidth": int(x.shape[3]) if x.ndim >= 4 else None,
            "YSamples": int(y.shape[0]) if y.ndim >= 1 else None,
            "YChannels": int(y.shape[1]) if y.ndim >= 2 else None,
            "YHeight": int(y.shape[2]) if y.ndim >= 3 else None,
            "YWidth": int(y.shape[3]) if y.ndim >= 4 else None,
            "XFiniteSampleCheck": bool(np.isfinite(np.asarray(x[:min(10, x.shape[0])])).all()),
            "YFiniteSampleCheck": bool(np.isfinite(np.asarray(y[:min(10, y.shape[0])])).all()),
            "MaskPositivePixels": int(np.asarray(y).sum()) if y.size < 250_000_000 else int(np.asarray(y[:min(128, y.shape[0])]).sum()),
        })

        if input_channels is not None and row["XChannels"] != int(input_channels):
            errors.append({
                "Scope": "Tensor",
                "Config": cfg,
                "Split": split,
                "Error": f"XChannels {row['XChannels']} != YAML InputChannels {input_channels}"
            })

        if len(features) and row["XChannels"] != len(features):
            errors.append({
                "Scope": "Tensor",
                "Config": cfg,
                "Split": split,
                "Error": f"XChannels {row['XChannels']} != len(Features) {len(features)}"
            })

        if cfg == "ConfigC" and row["XChannels"] == 12:
            # wind channels are last three by design
            sample = np.asarray(x[:min(32, x.shape[0])])
            wind_speed = sample[:, 9]
            wind_cos = sample[:, 10]
            wind_sin = sample[:, 11]
            row.update({
                "WindSpeedMin_sample": float(np.nanmin(wind_speed)),
                "WindSpeedMax_sample": float(np.nanmax(wind_speed)),
                "WindSpeedMean_sample": float(np.nanmean(wind_speed)),
                "WindCosMin_sample": float(np.nanmin(wind_cos)),
                "WindCosMax_sample": float(np.nanmax(wind_cos)),
                "WindSinMin_sample": float(np.nanmin(wind_sin)),
                "WindSinMax_sample": float(np.nanmax(wind_sin)),
            })

            if np.nanmin(wind_speed) < -1e-6:
                errors.append({"Scope": "Wind", "Config": cfg, "Split": split, "Error": "WindSpeed has negative values"})

            if np.nanmin(wind_cos) < -1.01 or np.nanmax(wind_cos) > 1.01:
                errors.append({"Scope": "Wind", "Config": cfg, "Split": split, "Error": "WindDirCos out of [-1,1]"})

            if np.nanmin(wind_sin) < -1.01 or np.nanmax(wind_sin) > 1.01:
                errors.append({"Scope": "Wind", "Config": cfg, "Split": split, "Error": "WindDirSin out of [-1,1]"})

        rows.append(row)

df = pd.DataFrame(rows)
out_dir = run_root / "Audit" / "LatestTables"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "TensorAudit.csv", index=False)

err_df = pd.DataFrame(errors)
err_df.to_csv(out_dir / "TensorAuditErrors.csv", index=False)

print("\nTENSOR AUDIT")
print(df.to_string(index=False))
print("\nERRORS")
print(err_df.to_string(index=False) if len(err_df) else "No tensor/config errors detected.")
print("\nSaved:", out_dir / "TensorAudit.csv")
print("Saved:", out_dir / "TensorAuditErrors.csv")
PY

Section "5. ModelFactory architecture audit"

python - <<'PY' | tee "${AuditDir}/ModelFactoryAudit.txt"
from pathlib import Path
import re

path = Path("Source/Models/ModelFactory.py")
text = path.read_text(encoding="utf-8")

print("ModelFactory path:", path)
print("")
print("Potential model names found:")

patterns = [
    r'["\']([A-Za-z0-9_]*UNet[A-Za-z0-9_]*)["\']',
    r'if\s+ModelName\s*==\s*["\']([^"\']+)["\']',
    r'elif\s+ModelName\s*==\s*["\']([^"\']+)["\']',
]

names = set()
for pat in patterns:
    names.update(re.findall(pat, text))

for name in sorted(names):
    print(" -", name)

print("")
print("CreateModel snippets:")
lines = text.splitlines()
for i, line in enumerate(lines, start=1):
    if "CreateModel" in line or "ModelName" in line or "SimpleUNet" in line or "EnhancedUNet" in line or "TransformerUNet" in line:
        lo = max(1, i-2)
        hi = min(len(lines), i+2)
        for j in range(lo, hi+1):
            print(f"{j:04d}: {lines[j-1]}")
        print("---")
PY

Section "6. Existing results schema and NaN audit"

python - <<'PY' | tee "${AuditDir}/ResultsNaNAudit.txt"
from pathlib import Path
import pandas as pd
import numpy as np

run_tag = "Exp271431"
root = Path("Outputs/Experiments") / run_tag

rows = []
details = []

for cfg in ["ConfigB", "ConfigC"]:
    cfg_root = root / cfg
    if not cfg_root.exists():
        continue

    for path in sorted(cfg_root.glob("*/Metrics/TestMetricsSummary.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            details.append({"Path": str(path), "Error": repr(exc)})
            continue

        row = {"FeatureConfigFolder": cfg, "Path": str(path), "Rows": len(df)}
        for col in [
            "FeatureConfig", "ModelName", "RunName", "Split", "Samples", "InputChannels",
            "MeanDice", "MeanIoU", "MeanPrecision", "MeanRecall",
            "GlobalDice", "GlobalIoU", "GlobalPrecision", "GlobalRecall"
        ]:
            row[col] = df[col].iloc[0] if col in df.columns and len(df) else np.nan

        missing_cols = [
            c for c in [
                "InputChannels",
                "GlobalPrecision",
                "GlobalRecall",
                "MeanPrecision",
                "MeanRecall",
            ]
            if c not in df.columns
        ]

        nan_cols = []
        for col in df.columns:
            if df[col].isna().any():
                nan_cols.append(col)

        row["MissingImportantColumns"] = ",".join(missing_cols)
        row["NaNColumns"] = ",".join(nan_cols)
        rows.append(row)

summary = pd.DataFrame(rows)
out_dir = root / "Audit" / "LatestTables"
out_dir.mkdir(parents=True, exist_ok=True)

if len(summary):
    summary.to_csv(out_dir / "ResultsSchemaNaNAudit.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSaved:", out_dir / "ResultsSchemaNaNAudit.csv")
else:
    print("No TestMetricsSummary.csv found.")

if details:
    pd.DataFrame(details).to_csv(out_dir / "ResultsReadErrors.csv", index=False)
    print("\nRead errors saved:", out_dir / "ResultsReadErrors.csv")
PY

Section "7. Checkpoint audit"

python - <<'PY' | tee "${AuditDir}/CheckpointAudit.txt"
from pathlib import Path
import pandas as pd
import torch

run_tag = "Exp271431"
root = Path("Outputs/Experiments") / run_tag

rows = []

for cfg in ["ConfigB", "ConfigC"]:
    cfg_root = root / cfg
    if not cfg_root.exists():
        continue

    for ckpt in sorted(cfg_root.glob("*/Checkpoints/BestModel.pt")):
        row = {
            "FeatureConfigFolder": cfg,
            "Checkpoint": str(ckpt),
            "Exists": ckpt.exists(),
        }

        try:
            payload = torch.load(ckpt, map_location="cpu")
            row["PayloadType"] = type(payload).__name__

            if isinstance(payload, dict):
                row["Keys"] = ",".join(sorted([str(k) for k in payload.keys()])[:30])
                row["InputChannels"] = payload.get("input_channels", payload.get("InputChannels", None))
                row["ModelName"] = payload.get("model_name", payload.get("ModelName", None))
                row["RunName"] = payload.get("run_name", payload.get("RunName", None))
                row["FeatureConfig"] = payload.get("feature_config", payload.get("FeatureConfig", None))
                row["Epoch"] = payload.get("epoch", None)
            else:
                row["Keys"] = ""
        except Exception as exc:
            row["Error"] = repr(exc)

        rows.append(row)

df = pd.DataFrame(rows)
out_dir = root / "Audit" / "LatestTables"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "CheckpointAudit.csv", index=False)

print(df.to_string(index=False) if len(df) else "No checkpoints found.")
print("\nSaved:", out_dir / "CheckpointAudit.csv")
PY

Section "8. Probability saturation quick audit"

python - <<'PY' | tee "${AuditDir}/ProbabilitySaturationAudit.txt"
from pathlib import Path
import pandas as pd
import numpy as np

run_tag = "Exp271431"
root = Path("Outputs/Experiments") / run_tag

rows = []

for cfg in ["ConfigB", "ConfigC"]:
    cfg_root = root / cfg
    if not cfg_root.exists():
        continue

    for path in sorted(cfg_root.glob("*/Metrics/TestMetricsBySample.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if len(df) == 0:
            continue

        run_dir = path.parents[1].name

        row = {
            "FeatureConfig": cfg,
            "RunDir": run_dir,
            "Samples": len(df),
            "MeanDice_mean": df["Dice"].mean() if "Dice" in df.columns else np.nan,
            "PredPixels_mean": df["PredictedPixels"].mean() if "PredictedPixels" in df.columns else np.nan,
            "PredPixels_p95": df["PredictedPixels"].quantile(0.95) if "PredictedPixels" in df.columns else np.nan,
            "PredPixels_max": df["PredictedPixels"].max() if "PredictedPixels" in df.columns else np.nan,
            "GTPixels_mean": df["GroundTruthPixels"].mean() if "GroundTruthPixels" in df.columns else np.nan,
            "ProbMean_mean": df["PredictionProbabilityMean"].mean() if "PredictionProbabilityMean" in df.columns else np.nan,
            "ProbMean_p95": df["PredictionProbabilityMean"].quantile(0.95) if "PredictionProbabilityMean" in df.columns else np.nan,
            "ProbMax_mean": df["PredictionProbabilityMax"].mean() if "PredictionProbabilityMax" in df.columns else np.nan,
        }

        if "PredictedPixels" in df.columns:
            row["SamplesPredOver10k"] = int((df["PredictedPixels"] > 10000).sum())
            row["SamplesPredOver20k"] = int((df["PredictedPixels"] > 20000).sum())

        rows.append(row)

out_dir = root / "Audit" / "LatestTables"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(rows)
df.to_csv(out_dir / "ProbabilitySaturationAudit.csv", index=False)

print(df.to_string(index=False) if len(df) else "No MetricsBySample files found.")
print("\nSaved:", out_dir / "ProbabilitySaturationAudit.csv")
PY

Section "9. Final audit summary"

python - <<'PY' | tee "${AuditDir}/FinalSummary.txt"
from pathlib import Path
import pandas as pd

run_tag = "Exp271431"
root = Path("Outputs/Experiments") / run_tag / "Audit" / "LatestTables"

print("Audit table directory:", root)

tables = [
    "TensorAudit.csv",
    "TensorAuditErrors.csv",
    "ResultsSchemaNaNAudit.csv",
    "CheckpointAudit.csv",
    "ProbabilitySaturationAudit.csv",
]

for t in tables:
    path = root / t
    print("")
    print("TABLE:", t)
    if not path.exists():
        print("  Missing")
        continue
    df = pd.read_csv(path)
    print("  Rows:", len(df))
    print("  Columns:", list(df.columns))
PY

AddReport ""
AddReport "## Output files"
AddReport ""
AddReport "- Console log: \`${Log}\`"
AddReport "- Risky hardcodes: \`${AuditDir}/RiskyHardcodes.txt\`"
AddReport "- Tensor audit: \`Outputs/Experiments/${RunTag}/Audit/LatestTables/TensorAudit.csv\`"
AddReport "- Tensor errors: \`Outputs/Experiments/${RunTag}/Audit/LatestTables/TensorAuditErrors.csv\`"
AddReport "- Results NaN/schema audit: \`Outputs/Experiments/${RunTag}/Audit/LatestTables/ResultsSchemaNaNAudit.csv\`"
AddReport "- Checkpoint audit: \`Outputs/Experiments/${RunTag}/Audit/LatestTables/CheckpointAudit.csv\`"
AddReport "- Probability saturation audit: \`Outputs/Experiments/${RunTag}/Audit/LatestTables/ProbabilitySaturationAudit.csv\`"

echo ""
echo "============================================================"
echo "AUDIT COMPLETED"
echo "============================================================"
echo "Report: ${Report}"
echo "Log: ${Log}"
echo "Latest tables: Outputs/Experiments/${RunTag}/Audit/LatestTables/"
