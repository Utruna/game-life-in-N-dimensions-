from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from engine import NDimLifeEngine, SimulationConfig
from rules import get_rule

State = np.ndarray | set[tuple[int, ...]]


def parse_shape(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("La taille doit contenir au moins 2 dimensions")
    if any(value <= 0 for value in values):
        raise ValueError("Toutes les dimensions doivent être > 0")
    return values


def init_random_dense(shape: tuple[int, ...], density: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8)


def resolve_run_backend(engine: NDimLifeEngine, initial_dense: np.ndarray) -> str:
    if engine.config.backend in {"dense", "sparse"}:
        return engine.config.backend
    live_ratio = float(np.count_nonzero(initial_dense)) / float(initial_dense.size if initial_dense.size else 1)
    return "sparse" if live_ratio < 0.10 else "dense"


def to_dense_state(engine: NDimLifeEngine, state: State) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state.astype(np.uint8)
    return engine.dense_from_coords(state)


def run_simulation(engine: NDimLifeEngine, initial_dense: np.ndarray, generations: int) -> list[np.ndarray]:
    runtime_backend = resolve_run_backend(engine, initial_dense)
    dense_states: list[np.ndarray] = [initial_dense.astype(np.uint8)]

    if runtime_backend == "dense":
        dense_state = initial_dense
        for _ in range(generations):
            dense_state = engine.step_dense(dense_state)
            dense_states.append(dense_state)
        return dense_states

    sparse_state = engine.sparse_from_dense(initial_dense)
    for _ in range(generations):
        sparse_state = engine.step_sparse(sparse_state)
        dense_states.append(to_dense_state(engine, sparse_state))
    return dense_states


def states_to_json_serializable(states: Iterable[np.ndarray]) -> list[Any]:
    return [state.tolist() for state in states]


def run() -> int:
    parser = argparse.ArgumentParser(description="Simulation headless Game of Life N-dimensions")
    parser.add_argument("--shape", required=True, help="Dimensions, ex: 20,20,20 ou 12,12,12,12")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--rules", default="life_3d_bays", help="Nom preset ou spec B.../S...")
    parser.add_argument("--backend", choices=["auto", "dense", "sparse"], default="auto")
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "hdf5"], default="json")
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    if not (0.0 <= args.density <= 1.0):
        raise ValueError("La densité doit être entre 0 et 1")

    config = SimulationConfig(shape=shape, rules=get_rule(args.rules), backend=args.backend)
    engine = NDimLifeEngine(config)

    initial = init_random_dense(shape, args.density, args.seed)
    states = run_simulation(engine, initial, args.generations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        payload = {
            "shape": list(shape),
            "rules": config.rules.spec,
            "generations": args.generations,
            "states": states_to_json_serializable(states),
        }
        args.output.write_text(json.dumps(payload), encoding="utf-8")
    else:
        with h5py.File(args.output, "w") as h5:
            h5.attrs["rules"] = config.rules.spec
            h5.attrs["shape"] = shape
            h5.attrs["generations"] = args.generations
            h5.create_dataset("states", data=np.stack(states, axis=0), compression="gzip")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
