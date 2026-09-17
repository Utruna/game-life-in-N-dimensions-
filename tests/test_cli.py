from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from cli.main import parse_shape, resolve_run_backend, run
from engine import NDimLifeEngine, RuleSet, SimulationConfig


def test_parse_shape_ok() -> None:
    assert parse_shape("10,11,12") == (10, 11, 12)


def test_parse_shape_invalid() -> None:
    with pytest.raises(ValueError):
        parse_shape("10")


def test_resolve_run_backend_auto() -> None:
    engine = NDimLifeEngine(SimulationConfig(shape=(4, 4, 4), rules=RuleSet.from_spec("B3/S2,3"), backend="auto"))
    low_density = np.zeros((4, 4, 4), dtype=np.uint8)
    low_density[0, 0, 0] = 1
    high_density = np.ones((4, 4, 4), dtype=np.uint8)

    assert resolve_run_backend(engine, low_density) == "sparse"
    assert resolve_run_backend(engine, high_density) == "dense"


def test_cli_invalid_density_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--shape", "6,6,6", "--density", "1.5", "--output", str(output)],
    )

    with pytest.raises(ValueError):
        run()


def test_cli_json_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--shape",
            "6,6,6",
            "--generations",
            "2",
            "--rules",
            "B6/S5,6,7",
            "--backend",
            "sparse",
            "--seed",
            "1",
            "--output",
            str(output),
            "--format",
            "json",
        ],
    )

    assert run() == 0
    payload = json.loads(output.read_text())
    assert payload["shape"] == [6, 6, 6]
    assert payload["generations"] == 2
    assert len(payload["states"]) == 3


def test_cli_hdf5_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.h5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--shape",
            "5,5,5,5",
            "--generations",
            "1",
            "--rules",
            "B18,19,20/S18,19,20,21",
            "--backend",
            "dense",
            "--seed",
            "2",
            "--output",
            str(output),
            "--format",
            "hdf5",
        ],
    )

    assert run() == 0
    with h5py.File(output, "r") as h5:
        assert tuple(h5.attrs["shape"]) == (5, 5, 5, 5)
        assert h5.attrs["rules"] == "B18,19,20/S18,19,20,21"
        assert int(h5.attrs["generations"]) == 1
        assert h5["states"].shape[0] == 2
