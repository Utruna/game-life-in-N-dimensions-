from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from engine import NDimLifeEngine, SimulationConfig
from rules import get_rule

app = FastAPI(title="N-Dimensional Game of Life")


def _parse_shape(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("VIZ_SHAPE doit contenir au moins 2 dimensions")
    if any(value <= 0 for value in values):
        raise ValueError("VIZ_SHAPE doit contenir des dimensions > 0")
    return values


SHAPE = _parse_shape(os.getenv("VIZ_SHAPE", "20,20,20"))
RULE_SPEC = os.getenv("VIZ_RULE", "life_3d_bays")
DENSITY = float(os.getenv("VIZ_DENSITY", "0.1"))
SEED = int(os.getenv("VIZ_SEED", "0"))

CONFIG = SimulationConfig(shape=SHAPE, rules=get_rule(RULE_SPEC), backend="dense")
ENGINE = NDimLifeEngine(CONFIG)
STATE = (np.random.default_rng(SEED).random(SHAPE) < DENSITY).astype(np.uint8)
STATE_LOCK = Lock()


@app.get("/")
def index() -> FileResponse:
    root = Path(__file__).resolve().parent
    return FileResponse(root / "static" / "index.html")


@app.get("/state")
def get_state(w: int = Query(default=0, ge=0)) -> dict[str, object]:
    with STATE_LOCK:
        snapshot = STATE.copy()

    if snapshot.ndim == 3:
        return {"ndim": 3, "shape": list(snapshot.shape), "live_cells": np.argwhere(snapshot > 0).tolist()}

    if snapshot.ndim == 4:
        slice_index = min(w, snapshot.shape[3] - 1)
        sliced = snapshot[:, :, :, slice_index]
        return {
            "ndim": 4,
            "shape": list(snapshot.shape),
            "slice_axis": 3,
            "slice_index": slice_index,
            "live_cells": np.argwhere(sliced > 0).tolist(),
        }

    return {"ndim": int(snapshot.ndim), "shape": list(snapshot.shape), "live_cells": []}


@app.post("/step")
def step() -> dict[str, object]:
    global STATE
    with STATE_LOCK:
        STATE = ENGINE.step_dense(STATE)
    return {"ok": True}
