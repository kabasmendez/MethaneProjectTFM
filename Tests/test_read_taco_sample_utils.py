import pandas as pd
import pytest

from Source.ReadTacoSample import FindSampleIndexById


def test_find_sample_index_by_id():
    table = pd.DataFrame({"id": ["A", "B", "C"]})
    assert FindSampleIndexById(table, "B") == 1


def test_find_sample_index_missing():
    table = pd.DataFrame({"id": ["A", "B", "C"]})
    with pytest.raises(ValueError):
        FindSampleIndexById(table, "Z")


def test_find_sample_index_duplicate():
    table = pd.DataFrame({"id": ["A", "B", "B"]})
    with pytest.raises(ValueError):
        FindSampleIndexById(table, "B")
