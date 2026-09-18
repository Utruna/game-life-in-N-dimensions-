from __future__ import annotations

import struct

import numpy as np
import pytest

from viz import server


HEADER_BYTES = 32


def unpack(payload: bytes) -> dict[str, int]:
    fields = struct.unpack(server.STATE_HEADER, payload[:HEADER_BYTES])
    keys = ("magic", "version", "sx", "sy", "sz", "sw", "generation", "live", "payload", "churn")
    return dict(zip(keys, fields))


@pytest.fixture(autouse=True)
def restore_server_state():
    """Le serveur garde son etat en variables de module : chaque test doit
    rendre la grille telle qu'il l'a trouvee, sinon l'ordre des tests compte."""
    saved = {
        name: getattr(server, name)
        for name in ("SHAPE", "MAX_NEIGHBORS", "CONFIG", "ENGINE", "RULE_SPEC",
                     "BOARD_SHAPE", "BOARD_SIZE", "BOARD_MINOR", "DENSITY", "SEED")
    }
    yield
    for name, value in saved.items():
        setattr(server, name, value)
    with server.STATE_LOCK:
        server._reseed_locked()


def use_grid(dim: int, size: int, shape: str = "cube", board: float | None = None) -> dict[str, object]:
    return server.set_board(shape=shape, size=float(size if board is None else board),
                            minor=float(max(2, size // 4)),
                            density=0.1, grid=size, dim=dim)


def test_3d_binary_header_keeps_sw_zero():
    """Le flux 3D doit rester octet pour octet ce qu'il etait : le champ libre
    devenu sw vaut 0, et un decodeur 3D existant lit la meme chose qu'avant."""
    use_grid(3, 20)
    head = unpack(server.get_state_binary().body)
    assert head["magic"] == server.STATE_MAGIC
    assert head["version"] == server.STATE_VERSION
    assert (head["sx"], head["sy"], head["sz"]) == (20, 20, 20)
    assert head["sw"] == 0


def test_4d_binary_header_carries_fourth_dimension():
    use_grid(4, 16)
    payload = server.get_state_binary().body
    head = unpack(payload)
    assert (head["sx"], head["sy"], head["sz"], head["sw"]) == (16, 16, 16, 16)
    # Masque de bits sur l'hypervolume entier, en ordre C.
    assert head["payload"] == 16**4 // 8
    assert len(payload) == HEADER_BYTES + head["payload"] + head["live"]


def test_4d_binary_mask_matches_state():
    """Le masque doit se re-deplier exactement sur la grille : c'est ce qui
    garantit que le client retrouve les bonnes coordonnees en 4D."""
    use_grid(4, 16)
    payload = server.get_state_binary().body
    head = unpack(payload)
    mask = np.frombuffer(payload, dtype=np.uint8, count=head["payload"], offset=HEADER_BYTES)
    alive = np.unpackbits(mask)[: server.STATE.size].reshape(server.SHAPE).astype(bool)
    assert np.array_equal(alive, server.STATE > 0)
    assert head["live"] == int(alive.sum())


def test_4d_volume_is_three_dimensional():
    """Une texture 3D ne peut pas recevoir quatre dimensions : /state/vol doit
    toujours annoncer et livrer un volume 3D, tranche ou projection."""
    use_grid(4, 16)
    for w in (-1, 0, 5):
        head = unpack(server.get_state_volume(w=w).body)
        assert (head["sx"], head["sy"], head["sz"]) == (16, 16, 16)
        assert head["sw"] == 0, "l'en-tete decrit le volume transmis, pas la grille"
        assert head["payload"] == 16**3


def test_4d_projection_counts_live_cells_along_w():
    use_grid(4, 16)
    body = server.get_state_volume(w=-1).body
    volume = np.frombuffer(body, dtype=np.uint8, offset=HEADER_BYTES).reshape((16, 16, 16))
    assert np.array_equal(volume, (server.STATE > 0).sum(axis=3).astype(np.uint8))


def test_4d_slice_matches_that_slice():
    use_grid(4, 16)
    body = server.get_state_volume(w=3).body
    volume = np.frombuffer(body, dtype=np.uint8, offset=HEADER_BYTES).reshape((16, 16, 16))
    expected = np.where(server.STATE[..., 3] > 0, np.minimum(server.AGES[..., 3], 254) + 1, 0)
    assert np.array_equal(volume, expected.astype(np.uint8))


def test_switching_dimension_switches_rule():
    """Garder une regle 3D face aux 80 voisins d'une grille 4D donne une grille
    morte au premier pas : la bascule doit reprendre la regle de la dimension."""
    use_grid(3, 20)
    assert server.RULE_SPEC == "life_3d_highlife"
    result = use_grid(4, 16)
    assert result["rule_name"] == server.DEFAULT_RULE_BY_DIM[4]
    assert server.CONFIG.rules.spec == server.get_rule("life_4d_chaos").spec


def test_4d_grid_is_capped_below_the_3d_limit():
    """160^4 ferait 655 millions de cases : le refus doit etre explicite, pas
    une allocation qui echoue."""
    with pytest.raises(server.HTTPException) as excinfo:
        server.set_board(shape="cube", size=60.0, density=0.1, grid=60, dim=4)
    assert excinfo.value.status_code == 400
    assert str(server.GRID_MAX_BY_DIM[4]) in excinfo.value.detail


def test_4d_rules_are_the_only_ones_offered_in_4d():
    use_grid(4, 16)
    names = {preset["name"] for preset in server.get_rules()["presets"]}
    assert "life_4d_chaos" in names
    assert not any(name.startswith("life_3d_") for name in names)


def test_4d_sphere_is_a_hypersphere_not_an_extrusion():
    """Prolonger la sphere 3D le long de w donnerait des tranches toutes
    identiques ; la vraie hypersphere a des tranches qui enflent puis se
    resorbent, et c'est ce qui rend la 4e dimension visible."""
    # Rayon nettement plus petit que la grille, sinon l'hypersphere la remplit
    # entierement et toutes les tranches se valent pour une autre raison.
    use_grid(4, 20, shape="sphere", board=7)
    mask = server.BOARD_MASK
    per_w = [int(mask[..., w].sum()) for w in range(mask.shape[3])]
    assert max(per_w) > 0
    assert len(set(per_w)) > 1, "des tranches identiques trahiraient une extrusion"
    middle = mask.shape[3] // 2
    assert per_w[middle] == max(per_w)
    assert per_w[0] < per_w[middle]


def test_4d_life_actually_survives():
    """La regle par defaut de la 4D doit rester vivante : c'est exactement le
    piege dans lequel tombe le preset historique life_4d_balanced."""
    use_grid(4, 20)
    for _ in range(25):
        server.step()
    assert int((server.STATE > 0).sum()) > 0


def _equilibrium(spec: str, generations: int = 120):
    """Remplissage et renouvellement a l'equilibre, les deux mesures sur
    lesquelles les commentaires de presets.py s'engagent."""
    from rules import get_rule
    from engine import NDimLifeEngine, SimulationConfig

    shape = (18,) * 4
    engine = NDimLifeEngine(
        SimulationConfig(shape=shape, rules=get_rule(spec), backend="dense")
    )
    grid = (np.random.default_rng(11).random(shape) < 0.1).astype(np.uint8)
    churn = counted = 0
    for step in range(generations):
        nxt = engine.step_dense(grid)
        if step >= generations - 20:
            churn += int(np.count_nonzero(nxt != grid))
            counted += 1
        grid = nxt
    live = int(grid.sum())
    assert live > 0, f"{spec} s'est eteinte"
    return live / grid.size, (churn / counted) / live


def test_calm_rule_is_really_calmer_than_chaos():
    """Le seul interet de life_4d_calm est d'etre moins agitee : si elle cesse
    de l'etre, son nom et son commentaire mentent."""
    _, chaos = _equilibrium("life_4d_chaos")
    fill, calm = _equilibrium("life_4d_calm")
    assert calm < chaos * 0.75, f"calm={calm:.0%} face a chaos={chaos:.0%}"
    assert fill > 0.10, "une regle qui se vide n'est plus calme, elle meurt"


def test_no_preset_is_a_still_life():
    """Le projet est parti d'une 3D qui figeait toujours sur le meme cube :
    un preset a renouvellement nul serait un retour en arriere."""
    for spec in ("life_4d_chaos", "life_4d_calm", "life_4d_mid"):
        _, renewal = _equilibrium(spec)
        assert renewal > 0.20, f"{spec} est quasi figee ({renewal:.0%})"


def test_documented_dead_rule_is_still_dead():
    """Si un jour cette regle devenait vivante, le commentaire qui la decrit
    serait faux : le test garde les deux en phase."""
    from rules import get_rule
    from engine import NDimLifeEngine, SimulationConfig

    shape = (16,) * 4
    engine = NDimLifeEngine(
        SimulationConfig(shape=shape, rules=get_rule("life_4d_balanced"), backend="dense")
    )
    grid = (np.random.default_rng(7).random(shape) < 0.1).astype(np.uint8)
    # Le commentaire annonce deux generations : on lui laisse une marge, mais
    # pas au point de ne plus rien verifier.
    for _ in range(4):
        grid = engine.step_dense(grid)
    assert int(grid.sum()) == 0
