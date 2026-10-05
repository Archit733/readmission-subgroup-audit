"""A fresh `git clone` must work no matter which folder the commands are run from."""
from pathlib import Path

import pytest

from readmit_audit.config import PROJECT_ROOT, load_config


def test_config_resolves_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                     # an unrelated folder, not the project root
    cfg = load_config()
    assert Path(cfg["data"]["path"]).is_absolute()
    assert Path(cfg["data"]["path"]).exists()
    assert Path(cfg["output"]["results_dir"]).is_absolute()
    assert str(PROJECT_ROOT) in cfg["output"]["results_dir"]


def test_relative_config_name_still_found_from_elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.yaml")
    assert Path(cfg["data"]["path"]).exists()


def test_missing_data_file_gives_clear_message(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("data:\n  path: data/nope.csv\noutput:\n  results_dir: out\n  figures_dir: figs\n")
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_config(str(bad))
