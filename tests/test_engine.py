import numpy as np

from engine import NDimLifeEngine, RuleSet, SimulationConfig


def test_neighbor_count_center_3d_is_26() -> None:
    config = SimulationConfig(shape=(3, 3, 3), rules=RuleSet.from_spec("B6/S567"))
    engine = NDimLifeEngine(config)
    grid = np.ones((3, 3, 3), dtype=np.uint8)
    counts = engine.count_neighbors_dense(grid)
    assert int(counts[1, 1, 1]) == 26


def test_neighbor_count_center_4d_is_80() -> None:
    config = SimulationConfig(shape=(3, 3, 3, 3), rules=RuleSet.from_spec("B18,19,20/S18,19,20"))
    engine = NDimLifeEngine(config)
    grid = np.ones((3, 3, 3, 3), dtype=np.uint8)
    counts = engine.count_neighbors_dense(grid)
    assert int(counts[1, 1, 1, 1]) == 80


def test_rules_birth_and_survival_applied() -> None:
    config = SimulationConfig(shape=(3, 3, 3), rules=RuleSet.from_spec("B3/S2,3"))
    engine = NDimLifeEngine(config)

    grid = np.zeros((3, 3, 3), dtype=np.uint8)
    grid[0, 1, 1] = 1
    grid[1, 0, 1] = 1
    grid[1, 1, 0] = 1

    next_grid = engine.step_dense(grid)
    assert int(next_grid[1, 1, 1]) == 1


def test_empty_grid_stays_empty() -> None:
    config = SimulationConfig(shape=(5, 5, 5), rules=RuleSet.from_spec("B6/S567"))
    engine = NDimLifeEngine(config)
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    next_grid = engine.step_dense(grid)
    assert np.count_nonzero(next_grid) == 0


def test_full_grid_with_no_survival_becomes_empty() -> None:
    config = SimulationConfig(shape=(4, 4, 4), rules=RuleSet.from_spec("B/S"))
    engine = NDimLifeEngine(config)
    grid = np.ones((4, 4, 4), dtype=np.uint8)
    next_grid = engine.step_dense(grid)
    assert np.count_nonzero(next_grid) == 0


def test_sparse_and_dense_step_consistency() -> None:
    config = SimulationConfig(shape=(4, 4, 4), rules=RuleSet.from_spec("B3/S2,3"))
    engine = NDimLifeEngine(config)
    coords = {(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)}

    dense = engine.dense_from_coords(coords)
    dense_next = engine.step_dense(dense)
    sparse_next = engine.step_sparse(coords)

    assert engine.sparse_from_dense(dense_next) == sparse_next
