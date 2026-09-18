from __future__ import annotations

import math
import os
import re
import struct
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from engine import NDimLifeEngine, SimulationConfig
from rules import PRESET_RULES, get_rule

app = FastAPI(title="N-Dimensional Game of Life")


def _parse_shape(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise ValueError("VIZ_SHAPE doit contenir au moins 2 dimensions")
    if any(value <= 0 for value in values):
        raise ValueError("VIZ_SHAPE doit contenir des dimensions > 0")
    return values


SHAPE = _parse_shape(os.getenv("VIZ_SHAPE", "20,20,20"))
MAX_NEIGHBORS = 3 ** len(SHAPE) - 1
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

# Changer de dimension sans changer de règle n'a pas de sens : les seuils de
# HighLife 3D (5 à 8 voisins) face aux 80 voisins d'une grille 4D donnent une
# grille qui meurt au premier pas. Chaque dimension a donc sa règle par défaut,
# appliquée automatiquement quand on bascule, sauf choix explicite de l'UI.
DEFAULT_RULE_BY_DIM = {3: "life_3d_highlife", 4: "life_4d_chaos"}
# 160³ tient (4,1 M cases) ; 160⁴ ferait 655 millions de cases. La 4ᵉ dimension
# s'achète en résolution : 48⁴ = 5,3 M cases, soit déjà plus qu'un 160³.
GRID_MAX_BY_DIM = {3: 160, 4: 48}
SUPPORTED_DIMS = (3, 4)
WORKERS = int(os.getenv("WEB_CONCURRENCY", os.getenv("VIZ_WORKERS", "1")))
if WORKERS != 1:
    raise RuntimeError("Le serveur viz en mémoire doit être lancé avec un seul worker")


def _random_seed() -> int:
    return int(np.random.SeedSequence().entropy % (2**32))


# La seed est une valeur persistante (visible/modifiable depuis l'UI), pas
# tirée à chaque reset : à seed égale, /reset rejoue exactement la même
# soupe initiale. VIZ_SEED permet de la figer dès le démarrage ; sinon on en
# tire une au hasard une seule fois.
_seed_env = os.getenv("VIZ_SEED")
SEED = int(_seed_env) if _seed_env is not None else _random_seed()

BOARD_SHAPES = {"cube", "sphere", "torus"}
# Forme de la zone dans laquelle la soupe initiale est semée : un cube de
# côté "size", une sphère de rayon "size", ou un tore de rayon "size" et de
# diamètre de tube "minor". En dehors de cette zone, les cellules restent
# mortes au démarrage.
BOARD_SHAPE = os.getenv("VIZ_BOARD", "cube")
if BOARD_SHAPE not in BOARD_SHAPES:
    raise ValueError(f"VIZ_BOARD doit être l'un de {sorted(BOARD_SHAPES)}")
BOARD_SIZE = float(os.getenv("VIZ_BOARD_SIZE", str(min(SHAPE))))
BOARD_MINOR = float(os.getenv("VIZ_BOARD_MINOR", str(max(1.0, min(SHAPE) / 4))))

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


def _board_mask() -> np.ndarray:
    ndim = len(SHAPE)
    centers = [(dim - 1) / 2 for dim in SHAPE]

    if BOARD_SHAPE == "cube":
        half = BOARD_SIZE / 2
        mask = np.ones(SHAPE, dtype=bool)
        for axis, dim in enumerate(SHAPE):
            axis_mask = np.abs(np.arange(dim) - centers[axis]) <= half
            mask &= axis_mask.reshape([dim if a == axis else 1 for a in range(ndim)])
        return mask

    if ndim < 3:
        raise ValueError("Les plateaux 'sphere'/'torus' nécessitent au moins 3 dimensions")

    coords = np.indices(SHAPE).astype(np.float64)
    delta = [coords[axis] - centers[axis] for axis in range(ndim)]
    tube_radius = BOARD_MINOR / 2

    if ndim == 3:
        dx, dy, dz = delta
        if BOARD_SHAPE == "sphere":
            return dx**2 + dy**2 + dz**2 <= BOARD_SIZE**2
        planar = np.sqrt(dx**2 + dy**2)
        return (planar - BOARD_SIZE) ** 2 + dz**2 <= tube_radius**2

    # En 4D on ne prolonge pas la forme 3D le long de w (ça donnerait un
    # hypercylindre, dont toutes les tranches en w sont identiques : la 4ᵉ
    # dimension n'apporterait alors rien à regarder). On prend les vrais
    # analogues 4D, dont les tranches en w évoluent.
    dx, dy, dz, dw = delta[:4]
    if BOARD_SHAPE == "sphere":
        # Hypersphère : ses tranches en w sont des sphères 3D qui enflent puis
        # se résorbent quand on traverse w.
        return dx**2 + dy**2 + dz**2 + dw**2 <= BOARD_SIZE**2
    # Tore de Clifford : deux rotations indépendantes dans les plans (x,y) et
    # (z,w), la seule surface qui soit « également tore » dans les deux.
    first = np.sqrt(dx**2 + dy**2) - BOARD_SIZE
    second = np.sqrt(dz**2 + dw**2) - BOARD_SIZE
    return first**2 + second**2 <= tube_radius**2


def _new_state() -> np.ndarray:
    grid = (np.random.default_rng(SEED).random(SHAPE) < DENSITY).astype(np.uint8)
    grid[~BOARD_MASK] = 0
    return grid


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


def _reseed_locked() -> None:
    """Repart de la génération 0. L'appelant doit détenir STATE_LOCK."""
    global STATE, AGES, GENERATION, BOARD_MASK, CHURN
    BOARD_MASK = _board_mask()
    STATE = _new_state()
    AGES = np.zeros(SHAPE, dtype=np.int32)
    GENERATION = 0
    CHURN = 0


GRID_MIN, GRID_MAX = 8, 160

# Flux binaire de /state/bin. En JSON, une grille 120³ pèse ~8,9 Mo et coûte
# 176 ms de sérialisation par étape ; en masque de bits c'est 216 ko et 0,1 ms,
# et le poids ne dépend plus que du volume, pas du nombre de cellules vivantes.
STATE_MAGIC = 0x4C494645  # 'LIFE'
STATE_VERSION = 1
# Le 4ᵉ champ court, resté libre en 3D, porte maintenant sw : la 4ᵉ dimension
# passe sans changer ni la taille ni la version de l'en-tête, et un flux 3D
# reste octet pour octet ce qu'il était (sw = 0 signifie « grille 3D »).
STATE_HEADER = "<IIHHHHIIII"  # magic, version, sx, sy, sz, sw, génération, vivantes, octets_utiles, renouvellement


def _pack_header(live_count: int, payload_bytes: int) -> bytes:
    sx, sy, sz = SHAPE[0], SHAPE[1], SHAPE[2]
    sw = SHAPE[3] if len(SHAPE) > 3 else 0
    return struct.pack(
        STATE_HEADER,
        STATE_MAGIC, STATE_VERSION,
        sx, sy, sz, sw,
        GENERATION, int(live_count), int(payload_bytes), CHURN,
    )


def _apply_grid_locked(size: int, ndim: int | None = None) -> None:
    """Redimensionne la grille (hypercubique). L'appelant doit détenir
    STATE_LOCK et appeler _reseed_locked() ensuite."""
    global SHAPE, MAX_NEIGHBORS, CONFIG, ENGINE, RULE_SPEC
    ndim = len(SHAPE) if ndim is None else ndim
    rules = CONFIG.rules
    if ndim != len(SHAPE):
        # La règle courante est calibrée pour l'ancien voisinage ; la garder en
        # changeant de dimension donne une grille morte au premier pas.
        RULE_SPEC = DEFAULT_RULE_BY_DIM[ndim]
        rules = get_rule(RULE_SPEC)
    SHAPE = (size,) * ndim
    MAX_NEIGHBORS = 3 ** len(SHAPE) - 1
    CONFIG = SimulationConfig(shape=SHAPE, rules=rules, backend="dense")
    ENGINE = NDimLifeEngine(CONFIG)


def _board_capacity() -> int:
    return int(BOARD_MASK.sum())


def _board_hint() -> str | None:
    """Un plateau hors grille ne donne jamais aucune cellule, quelle que soit la
    seed : on dit pourquoi, et quelle taille de grille il faudrait."""
    if _board_capacity() > 0:
        return None
    exponent = "³" if len(SHAPE) == 3 else "⁴"
    if BOARD_SHAPE == "torus":
        needed = math.ceil(2 * (BOARD_SIZE + BOARD_MINOR / 2) + 1)
        reach = math.hypot((SHAPE[0] - 1) / 2, (SHAPE[1] - 1) / 2)
        ceiling = GRID_MAX_BY_DIM[len(SHAPE)]
        note = (
            ""
            if needed <= ceiling
            else f" Or une grille {len(SHAPE)}D est plafonnée à {ceiling} : réduisez plutôt le tore."
        )
        return (
            f"Ce tore est hors de la grille {SHAPE[0]}{exponent} : l'anneau commence à "
            f"{BOARD_SIZE - BOARD_MINOR / 2:.1f} du centre alors que la grille ne porte que "
            f"jusqu'à {reach:.1f}. Il faut une grille d'au moins {needed}{exponent}.{note}"
        )
    return f"Ce plateau ne contient aucune cellule dans une grille {SHAPE[0]}{exponent}."


_DIM_TAG = re.compile(r"_(\d+)d_")


def _rule_fits(rule, name: str = "") -> bool:
    """Écarte les règles inapplicables ici : seuils de voisinage hors d'atteinte,
    ou preset explicitement prévu pour une autre dimension."""
    values = rule.birth | rule.survive
    if values and max(values) > MAX_NEIGHBORS:
        return False
    tag = _DIM_TAG.search(name)
    return tag is None or int(tag.group(1)) == len(SHAPE)


# Le plateau est le terrain de jeu, pas seulement la mise en place : le masque
# s'applique aussi à chaque génération. Sans ça une règle expansive comme
# HighLife ressort du tore en une vingtaine d'étapes et la forme choisie
# disparaît aussitôt.
BOARD_MASK = _board_mask()
STATE = _new_state()
# L'âge d'une cellule (nombre de générations survécues) est ce qui permet au
# rendu de distinguer une naissance d'une structure installée : sans lui, une
# soupe chaotique n'est qu'un nuage uniforme qui clignote.
AGES = np.zeros(SHAPE, dtype=np.int32)
GENERATION = 0
CHURN = 0


@app.get("/")
def index() -> FileResponse:
    root = Path(__file__).resolve().parent
    return FileResponse(root / "static" / "index.html")


@app.get("/state")
def get_state(w: int = Query(default=0, ge=0)) -> dict[str, object]:
    with STATE_LOCK:
        snapshot = STATE.copy()
        ages = AGES.copy()
        generation = GENERATION

    if snapshot.ndim == 3:
        alive = snapshot > 0
        return {
            "ndim": 3,
            "shape": list(snapshot.shape),
            "live_cells": np.argwhere(alive).tolist(),
            "ages": ages[alive].tolist(),
            "seed": SEED,
            "rule": CONFIG.rules.spec,
            "rule_name": RULE_SPEC,
            "generation": generation,
        }

    if snapshot.ndim == 4:
        slice_index = min(w, snapshot.shape[3] - 1)
        sliced = snapshot[:, :, :, slice_index]
        return {
            "ndim": 4,
            "shape": list(snapshot.shape),
            "slice_axis": 3,
            "slice_index": slice_index,
            "slice_live_cells": np.argwhere(sliced > 0).tolist(),
            # La grille 4D complète ne repasse pas en JSON : à 48⁴ la liste des
            # cellules vivantes pèse des dizaines de Mo et des secondes de
            # sérialisation. C'est /state/bin qui la porte.
            "live_count": int((snapshot > 0).sum()),
            "seed": SEED,
            "rule": CONFIG.rules.spec,
            "rule_name": RULE_SPEC,
            "generation": generation,
        }

    return {
        "ndim": int(snapshot.ndim),
        "shape": list(snapshot.shape),
        "live_cells": [],
        "seed": SEED,
        "generation": generation,
    }


@app.post("/step")
def step() -> dict[str, object]:
    global STATE, AGES, GENERATION, CHURN
    with STATE_LOCK:
        nxt = _apply_noise(ENGINE.step_dense(STATE))
        nxt[~BOARD_MASK] = 0
        # Survivante -> age+1 ; nouvelle-née (y compris via météore) -> 0.
        AGES = np.where(nxt > 0, np.where(STATE > 0, AGES + 1, 0), 0)
        # Le client ne peut plus déduire le renouvellement en comparant deux
        # états (il n'en garde plus la liste en mode rapide) : on le calcule ici.
        CHURN = int(np.count_nonzero(nxt != STATE))
        STATE = nxt
        GENERATION += 1
        generation = GENERATION
    return {"ok": True, "generation": generation}


@app.get("/meta")
def get_meta() -> dict[str, object]:
    """Métadonnées seules : le flux binaire porte la grille, pas le contexte."""
    return {
        "ndim": len(SHAPE),
        "shape": list(SHAPE),
        "seed": SEED,
        "rule": CONFIG.rules.spec,
        "rule_name": RULE_SPEC,
        "generation": GENERATION,
    }


@app.get("/state/vol")
def get_state_volume(w: int = Query(default=-1)) -> Response:
    """Volume dense 3D, un octet par case, prêt à être poussé tel quel dans une
    texture 3D. Le client n'a rien à décoder — c'est une recopie vers le GPU.

    En 3D : 0 = morte, 1..255 = âge+1.
    En 4D il faut bien ramener quatre dimensions à trois, et l'en-tête annonce
    donc toujours une grille 3D :
      * w >= 0 : la tranche w, avec la même convention d'âge qu'en 3D ;
      * w < 0  : la projection, où l'octet compte les cases vivantes le long de
        w. Ce n'est plus un âge mais une épaisseur : chaque voxel dit combien de
        la 4ᵉ dimension est occupée au-dessus de lui, ce que le raymarcher rend
        comme une densité. Le client sait ce qu'il a demandé et choisit la
        rampe correspondante.
    """
    with STATE_LOCK:
        if len(SHAPE) == 3:
            alive = STATE > 0
            volume = np.where(alive, np.minimum(AGES, 254) + 1, 0).astype(np.uint8)
            live_count = int(alive.sum())
            header = _pack_header(live_count, volume.size)
        elif len(SHAPE) == 4:
            live_count = int((STATE > 0).sum())
            if w >= 0:
                index = min(w, SHAPE[3] - 1)
                sliced, sliced_ages = STATE[..., index], AGES[..., index]
                volume = np.where(
                    sliced > 0, np.minimum(sliced_ages, 254) + 1, 0
                ).astype(np.uint8)
            else:
                volume = np.minimum((STATE > 0).sum(axis=3), 255).astype(np.uint8)
            # L'en-tête décrit le volume transmis, pas la grille : le client
            # reçoit bien un 3D à charger dans la texture.
            header = struct.pack(
                STATE_HEADER,
                STATE_MAGIC, STATE_VERSION,
                SHAPE[0], SHAPE[1], SHAPE[2], 0,
                GENERATION, live_count, int(volume.size), CHURN,
            )
        else:
            raise HTTPException(
                status_code=409,
                detail="Le flux volumétrique attend une grille 3D ou 4D",
            )
    return Response(content=header + volume.tobytes(),
                    media_type="application/octet-stream")


@app.get("/state/bin")
def get_state_binary() -> Response:
    """Masque de bits + âges, en ordre C. Indépendant du nombre de dimensions :
    l'en-tête porte la forme, et le client en déduit le découpage des indices."""
    if len(SHAPE) not in SUPPORTED_DIMS:
        raise HTTPException(
            status_code=409, detail="Le flux binaire attend une grille 3D ou 4D"
        )
    with STATE_LOCK:
        alive = STATE > 0
        mask = np.packbits(alive)  # aplati en ordre C, bit de poids fort en tête
        ages = np.minimum(AGES[alive], 255).astype(np.uint8)
        header = _pack_header(ages.size, mask.size)
    return Response(content=header + mask.tobytes() + ages.tobytes(),
                    media_type="application/octet-stream")


@app.post("/reset")
def reset() -> dict[str, object]:
    with STATE_LOCK:
        _reseed_locked()
    return {"ok": True, "seed": SEED, "generation": GENERATION}


@app.get("/seed")
def get_seed() -> dict[str, object]:
    return {"seed": SEED}


@app.post("/seed")
def set_seed(value: int = Query(...)) -> dict[str, object]:
    global SEED
    with STATE_LOCK:
        SEED = value
        _reseed_locked()
    return {"seed": SEED, "generation": GENERATION}


@app.post("/seed/random")
def randomize_seed() -> dict[str, object]:
    global SEED
    with STATE_LOCK:
        SEED = _random_seed()
        _reseed_locked()
    return {"seed": SEED, "generation": GENERATION}


@app.get("/board")
def get_board() -> dict[str, object]:
    return {
        "shape": BOARD_SHAPE,
        "size": BOARD_SIZE,
        "minor": BOARD_MINOR,
        "density": DENSITY,
        "shapes": sorted(BOARD_SHAPES),
        "grid": SHAPE[0],
        "grid_min": GRID_MIN,
        "grid_max": GRID_MAX_BY_DIM[len(SHAPE)],
        "dim": len(SHAPE),
        "dims": list(SUPPORTED_DIMS),
        "grid_max_by_dim": GRID_MAX_BY_DIM,
        "capacity": _board_capacity(),
        "hint": _board_hint(),
    }


@app.post("/board")
def set_board(
    shape: str = Query(...),
    size: float = Query(..., gt=0),
    minor: float | None = Query(default=None, gt=0),
    density: float | None = Query(default=None, gt=0.0, le=1.0),
    grid: int | None = Query(default=None, ge=GRID_MIN, le=GRID_MAX),
    dim: int | None = Query(default=None),
) -> dict[str, object]:
    global BOARD_SHAPE, BOARD_SIZE, BOARD_MINOR, DENSITY
    if shape not in BOARD_SHAPES:
        raise HTTPException(status_code=400, detail=f"shape doit être l'un de {sorted(BOARD_SHAPES)}")
    if dim is not None and dim not in SUPPORTED_DIMS:
        raise HTTPException(status_code=400, detail=f"dim doit être l'un de {list(SUPPORTED_DIMS)}")
    target_dim = len(SHAPE) if dim is None else dim
    target_grid = SHAPE[0] if grid is None else grid
    # Le plafond dépend de la dimension : ce qui passe en 3D fait exploser la
    # mémoire en 4D (160⁴ = 655 M de cases), donc on refuse explicitement au
    # lieu de laisser l'allocation échouer.
    ceiling = GRID_MAX_BY_DIM[target_dim]
    if target_grid > ceiling:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Une grille {target_dim}D est limitée à {ceiling} de côté "
                f"({ceiling ** target_dim:,} cases) ; {target_grid} demandé."
            ),
        )
    with STATE_LOCK:
        # Grille et plateau changent ensemble : sinon, agrandir un tore avant
        # d'agrandir la grille donne un état vide intermédiaire.
        if target_grid != SHAPE[0] or target_dim != len(SHAPE):
            _apply_grid_locked(target_grid, target_dim)
        BOARD_SHAPE = shape
        BOARD_SIZE = size
        if minor is not None:
            BOARD_MINOR = minor
        if density is not None:
            DENSITY = density
        _reseed_locked()
        capacity = _board_capacity()
        hint = _board_hint()
    return {
        "shape": BOARD_SHAPE,
        "size": BOARD_SIZE,
        "minor": BOARD_MINOR,
        "density": DENSITY,
        "grid": SHAPE[0],
        "dim": len(SHAPE),
        "grid_max": GRID_MAX_BY_DIM[len(SHAPE)],
        # Changer de dimension change la règle d'office : le client doit le
        # savoir pour ne pas afficher l'ancienne.
        "rule_name": RULE_SPEC,
        "rule": CONFIG.rules.spec,
        "capacity": capacity,
        "hint": hint,
        "seed": SEED,
        "generation": GENERATION,
    }


@app.get("/rules")
def get_rules() -> dict[str, object]:
    return {
        "current": RULE_SPEC,
        "spec": CONFIG.rules.spec,
        "presets": [
            {"name": name, "spec": rule.spec}
            for name, rule in PRESET_RULES.items()
            if _rule_fits(rule, name)
        ],
    }


@app.post("/rule")
def set_rule(spec: str = Query(...)) -> dict[str, object]:
    global RULE_SPEC, CONFIG, ENGINE
    try:
        rules = get_rule(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not _rule_fits(rules):
        raise HTTPException(
            status_code=400,
            detail=f"Règle incompatible avec une grille {len(SHAPE)}D (max {MAX_NEIGHBORS} voisins)",
        )
    with STATE_LOCK:
        RULE_SPEC = spec
        CONFIG = SimulationConfig(shape=SHAPE, rules=rules, backend="dense")
        ENGINE = NDimLifeEngine(CONFIG)
        _reseed_locked()
    return {"current": RULE_SPEC, "spec": CONFIG.rules.spec, "generation": GENERATION}


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
