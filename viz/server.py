from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from engine import NDimLifeEngine, SimulationConfig
from rules import get_rule

app = FastAPI(title="N-Dimensional Game of Life")

SHAPE = (20, 20, 20)
CONFIG = SimulationConfig(shape=SHAPE, rules=get_rule("life_3d_bays"), backend="dense")
ENGINE = NDimLifeEngine(CONFIG)
STATE = (np.random.default_rng(0).random(SHAPE) < 0.1).astype(np.uint8)
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
