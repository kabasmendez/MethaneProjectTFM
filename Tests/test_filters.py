import pandas as pd

from Source.Filters import ApplyMandatoryFilters, DetectFilterColumns


def test_detect_filter_columns():
    df = pd.DataFrame(
        {
            "detection:sector": ["Oil and Gas"],
            "quality:percentage_clear": [95.0],
            "quality:observability": ["clear"],
            "detection:isplume": [True],
            "plume:geometry": ["geom"],
            "HasTarget": [True],
            "HasReference": [True],
            "HasPlume": [True],
        }
    )

    table, detected = DetectFilterColumns(df)

    assert detected["Sector"] == "detection:sector"
    assert detected["PercentageClear"] == "quality:percentage_clear"
    assert len(table) > 0


def test_apply_mandatory_filters_keeps_valid_sample():
    df = pd.DataFrame(
        {
            "SampleId": ["A", "B"],
            "detection:sector": ["Oil and Gas", "Coal"],
            "quality:percentage_clear": [95.0, 95.0],
            "quality:observability": ["clear", "clear"],
            "detection:isplume": [True, True],
            "plume:geometry": ["geom", "geom"],
            "HasTarget": [True, True],
            "HasReference": [True, True],
            "HasPlume": [True, True],
        }
    )

    config = {
        "Filters": {
            "Sector": "Oil and Gas",
            "MinPercentageClear": 90.0,
            "Observability": "clear",
            "RequirePlume": True,
            "RequireGeometry": True,
            "RequireProducts": ["target", "reference", "plume"],
        }
    }

    filtered, summary, detected = ApplyMandatoryFilters(df, config)

    assert len(filtered) == 1
    assert filtered.iloc[0]["SampleId"] == "A"
    assert len(summary) > 0
    assert len(detected) > 0
