# Game of Life en N dimensions (3D/4D)

Projet Python pour simuler un Jeu de la Vie généralisé en N dimensions, avec prise en charge explicite des cas 3D (26 voisins) et 4D (80 voisins).

## Structure

- `engine/` : moteur de simulation pur, générique N-D
- `rules/` : presets de règles et résolution de formats `B.../S...`
- `cli/` : exécution headless et export JSON/HDF5
- `viz/` : serveur FastAPI pour exposer l'état de grille en JSON (base pour rendu Three.js)
- `tests/` : tests unitaires pytest

## Règles B/S

Le format suit `B.../S...` :
- `B` (birth) : nombres de voisins qui font naître une cellule morte
- `S` (survive) : nombres de voisins qui maintiennent une cellule vivante

Exemples :
- `B3/S2,3`
- `B5,6,7/S6,7,8`

## 3D vs 4D

- En 3D, une cellule a `3^3 - 1 = 26` voisins.
- En 4D, une cellule a `3^4 - 1 = 80` voisins.

Le moteur n'est pas codé en dur pour 3D/4D : il fonctionne pour toute dimension N via un noyau de convolution généré dynamiquement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

Les tests couvrent notamment :
- comptage de voisins en 3D/4D,
- application des règles de naissance/survie,
- cas limites (grille vide, grille pleine),
- cohérence dense/sparse.

## Lancer une simulation headless

```bash
python -m cli.main \
  --shape 20,20,20 \
  --generations 10 \
  --rules life_3d_bays \
  --backend auto \
  --density 0.1 \
  --seed 42 \
  --format json \
  --output outputs/run.json
```

Pour du 4D, par exemple : `--shape 12,12,12,12`.

Formats d'export :
- `json` : état complet par génération
- `hdf5` : dataset compressé `states`

## Presets

Dans `rules/presets.py` :
- `life_3d_bays` : inspiré des travaux de Carter Bays (exploration 3D)
- `life_3d_highlife`
- `life_4d_balanced`

## Visualisation (base)

Le module `viz/server.py` expose :
- `GET /state` : cellules vivantes en JSON
- `POST /step` : avance d'une génération

Pour la 4D, l'API accepte un index de tranche (`w`) pour visualiser une coupe 3D de la 4e dimension.

Démarrage :

```bash
uvicorn viz.server:app --reload --workers 1
```
