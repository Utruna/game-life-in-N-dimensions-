from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.ndimage import convolve

from engine.rules import RuleSet

Coord = tuple[int, ...]


@dataclass
class SimulationConfig:
    shape: tuple[int, ...]
    rules: RuleSet
    backend: str = "auto"  # auto|dense|sparse


class NDimLifeEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.ndim = len(config.shape)
        self._kernel = np.ones((3,) * self.ndim, dtype=np.int8)
        self._kernel[(1,) * self.ndim] = 0
        self._neighbor_offsets = [
            offset
            for offset in product((-1, 0, 1), repeat=self.ndim)
            if any(component != 0 for component in offset)
        ]

    def empty_dense(self) -> np.ndarray:
        return np.zeros(self.config.shape, dtype=np.uint8)

    def dense_from_coords(self, coords: Iterable[Coord]) -> np.ndarray:
        grid = self.empty_dense()
        for coord in coords:
            if self._in_bounds(coord):
                grid[coord] = 1
        return grid

    def sparse_from_dense(self, grid: np.ndarray) -> set[Coord]:
        live = np.argwhere(grid > 0)
        return {tuple(int(v) for v in coord) for coord in live}

    def step_dense(self, grid: np.ndarray) -> np.ndarray:
        if grid.ndim != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {grid.ndim}")

        neighbors = convolve(grid.astype(np.int16), self._kernel, mode="constant", cval=0)
        birth_mask = np.isin(neighbors, tuple(self.config.rules.birth)) & (grid == 0)
        survive_mask = np.isin(neighbors, tuple(self.config.rules.survive)) & (grid == 1)
        next_grid = np.zeros_like(grid, dtype=np.uint8)
        next_grid[birth_mask | survive_mask] = 1
        return next_grid

    def step_sparse(self, live_cells: set[Coord]) -> set[Coord]:
        neighbor_counts: dict[Coord, int] = defaultdict(int)

        for coord in live_cells:
            for offset in self._neighbor_offsets:
                neighbor = tuple(coord[axis] + offset[axis] for axis in range(self.ndim))
                if self._in_bounds(neighbor):
                    neighbor_counts[neighbor] += 1

        next_cells: set[Coord] = set()
        for coord, count in neighbor_counts.items():
            alive = coord in live_cells
            if not alive and count in self.config.rules.birth:
                next_cells.add(coord)
            elif alive and count in self.config.rules.survive:
                next_cells.add(coord)

        return next_cells

    def count_neighbors_dense(self, grid: np.ndarray) -> np.ndarray:
        return convolve(grid.astype(np.int16), self._kernel, mode="constant", cval=0)

    def run_dense(self, grid: np.ndarray, generations: int) -> list[np.ndarray]:
        states = [grid.astype(np.uint8)]
        current = states[0]
        for _ in range(generations):
            current = self.step_dense(current)
            states.append(current)
        return states

    def run_sparse(self, live_cells: set[Coord], generations: int) -> list[set[Coord]]:
        states = [set(live_cells)]
        current = states[0]
        for _ in range(generations):
            current = self.step_sparse(current)
            states.append(current)
        return states

    def step(self, state: np.ndarray | set[Coord]) -> np.ndarray | set[Coord]:
        backend = self._resolve_backend(state)
        if backend == "dense":
            if not isinstance(state, np.ndarray):
                state = self.dense_from_coords(state)
            return self.step_dense(state)
        if not isinstance(state, set):
            state = self.sparse_from_dense(state)
        return self.step_sparse(state)

    def _resolve_backend(self, state: np.ndarray | set[Coord]) -> str:
        if self.config.backend in {"dense", "sparse"}:
            return self.config.backend
        if isinstance(state, set):
            return "sparse"
        live_ratio = float(np.count_nonzero(state)) / float(state.size if state.size else 1)
        return "sparse" if live_ratio < 0.10 else "dense"

    def _in_bounds(self, coord: Coord) -> bool:
        return all(0 <= coord[idx] < self.config.shape[idx] for idx in range(self.ndim))
