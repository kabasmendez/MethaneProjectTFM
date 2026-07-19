"""
AuditUtils.py

Utilidades para auditoría mínima reproducible.

Cada script importante debe guardar un JSON con:
- ScriptName
- RunTag
- fecha UTC
- parámetros
- entradas
- salidas
- estado
- detalles técnicos relevantes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def GetUtcTimestamp() -> str:
    """Devuelve timestamp UTC ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def ToJsonable(Value: Any) -> Any:
    """Convierte objetos comunes a tipos serializables."""
    if isinstance(Value, Path):
        return str(Value)

    if isinstance(Value, dict):
        return {str(Key): ToJsonable(Item) for Key, Item in Value.items()}

    if isinstance(Value, (list, tuple, set)):
        return [ToJsonable(Item) for Item in Value]

    return Value


def WriteJson(Data: dict[str, Any], OutputPath: Path) -> None:
    """Guarda un diccionario como JSON."""
    OutputPath.parent.mkdir(parents=True, exist_ok=True)

    with OutputPath.open("w", encoding="utf-8") as File:
        json.dump(ToJsonable(Data), File, indent=4, ensure_ascii=False)


def BuildAuditRecord(
    ScriptName: str,
    RunTag: str,
    Parameters: dict[str, Any],
    Inputs: dict[str, Any],
    Outputs: dict[str, Any],
    Status: str,
    Details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye registro estándar de auditoría."""
    return {
        "ScriptName": ScriptName,
        "RunTag": RunTag,
        "TimestampUtc": GetUtcTimestamp(),
        "Parameters": Parameters,
        "Inputs": Inputs,
        "Outputs": Outputs,
        "Status": Status,
        "Details": Details or {},
    }


def AppendOutputIndex(
    OutputIndexPath: Path,
    RunTag: str,
    Step: str,
    Config: str,
    Model: str,
    OutputType: str,
    RelativePath: str,
    Created: bool,
    Description: str,
) -> None:
    """Añade una fila a Audit/OutputIndex.csv."""
    OutputIndexPath.parent.mkdir(parents=True, exist_ok=True)

    Header = "RunTag,Step,Config,Model,OutputType,RelativePath,Created,Description\n"
    Row = (
        f"{RunTag},{Step},{Config},{Model},{OutputType},"
        f"{RelativePath},{str(Created).lower()},{Description}\n"
    )

    if not OutputIndexPath.exists():
        OutputIndexPath.write_text(Header, encoding="utf-8")

    with OutputIndexPath.open("a", encoding="utf-8") as File:
        File.write(Row)
