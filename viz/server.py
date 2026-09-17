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
# La règle 5766 (B6/S5,6,7, ex-défaut) converge très vite vers des natures
# mortes (le fameux cube 2x2x2). Les règles "historiques" de Bays (4555,
# 5655, 6855) ont été testées avec nos bords non-toroïdaux : elles s'éteignent
# quasiment toujours en une trentaine d'étapes à partir d'une soupe aléatoire.
# La variante "HighLife" 3D (B5,6,7/S6,7,8) est la seule à rester en équilibre
# chaotique durable (~30% de cellules vivantes, avec un fort taux de
# renouvellement à chaque étape) : c'est elle qui donne le rendu le plus
# "vivant" en continu.
RULE_SPEC = os.getenv("VIZ_RULE", "life_3d_highlife")
DENSITY = float(os.getenv("VIZ_DENSITY", "0.1"))
SEED = int(os.getenv("VIZ_SEED", "0"))
WORKERS = int(os.getenv("WEB_CONCURRENCY", os.getenv("VIZ_WORKERS", "1")))
if WORKERS != 1:
    raise RuntimeError("Le serveur viz en mémoire doit être lancé avec un seul worker")

# Un seul cellule isolée injectée aléatoirement meurt quasi toujours au tour
# suivant (elle n'a aucun voisin) : ça ne fait que scintiller sans rien
# produire. On simule plutôt des "impacts de météore" : un petit amas de
# cellules est déposé à un endroit aléatoire, avec assez de densité pour que
# certaines sous-structures survivent et interagissent avec le reste de la
# grille au lieu de disparaître immédiatement.
NOISE_RATE = float(os.getenv("VIZ_NOISE", "0.12"))  # probabilité d'un impact par étape
NOISE_ENABLED = os.getenv("VIZ_NOISE_ENABLED", "1") != "0"
METEOR_SIZE = int(os.getenv("VIZ_METEOR_SIZE", "3"))
METEOR_DENSITY = float(os.getenv("VIZ_METEOR_DENSITY", "0.55"))

CONFIG = SimulationConfig(shape=SHAPE, rules=get_rule(RULE_SPEC), backend="dense")
ENGINE = NDimLifeEngine(CONFIG)
STATE_LOCK = Lock()
NOISE_RNG = np.random.default_rng()


def _new_state() -> np.ndarray:
    return (np.random.default_rng(SEED).random(SHAPE) < DENSITY).astype(np.uint8)


def _apply_noise(grid: np.ndarray) -> np.ndarray:
    if not NOISE_ENABLED or NOISE_RATE <= 0:
        return grid
    if NOISE_RNG.random() >= NOISE_RATE:
        return grid

    grid = grid.copy()
    sizes = [min(METEOR_SIZE, dim) for dim in grid.shape]
    origin = [int(NOISE_RNG.integers(0, dim - size + 1)) for dim, size in zip(grid.shape, sizes)]
    region = tuple(slice(o, o + size) for o, size in zip(origin, sizes))
    patch = grid[region]
    patch[NOISE_RNG.random(patch.shape) < METEOR_DENSITY] = 1
    return grid


STATE = _new_state()


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
            "slice_live_cells": np.argwhere(sliced > 0).tolist(),
            "live_cells_4d": np.argwhere(snapshot > 0).tolist(),
        }

    return {"ndim": int(snapshot.ndim), "shape": list(snapshot.shape), "live_cells": []}


@app.post("/step")
def step() -> dict[str, object]:
    global STATE
    with STATE_LOCK:
        STATE = _apply_noise(ENGINE.step_dense(STATE))
    return {"ok": True}


@app.post("/reset")
def reset() -> dict[str, object]:
    global STATE
    with STATE_LOCK:
        STATE = _new_state()
    return {"ok": True}


@app.get("/noise")
def get_noise() -> dict[str, object]:
    return {"enabled": NOISE_ENABLED, "rate": NOISE_RATE}


@app.post("/noise")
def set_noise(
    enabled: bool = Query(default=True),
    rate: float = Query(default=0.12, ge=0.0, le=1.0),
) -> dict[str, object]:
    global NOISE_ENABLED, NOISE_RATE
    with STATE_LOCK:
        NOISE_ENABLED = enabled
        NOISE_RATE = rate
    return {"enabled": NOISE_ENABLED, "rate": NOISE_RATE}
