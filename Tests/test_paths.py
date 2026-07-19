from pathlib import Path

import pytest

from Source.Paths import GetProjectRoot, ValidateRunTag


def test_valid_run_tag():
    assert ValidateRunTag("Exp241930") == "Exp241930"


def test_invalid_run_tag():
    with pytest.raises(ValueError):
        ValidateRunTag("Experiment241930")


def test_project_root_exists():
    assert Path(GetProjectRoot()).exists()
