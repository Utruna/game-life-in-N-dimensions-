from __future__ import annotations

from engine.rules import RuleSet

# Presets 3D issus de la littérature sur le Life 3D (notamment Carter Bays).
# Les règles exactes dépendent de la variante; ces presets servent de base d'exploration.
PRESET_RULES: dict[str, RuleSet] = {
    # Life 5766 (Bays 1987) : stable, converge vite vers des natures mortes
    # (ex: cube 2x2x2) a partir d'une soupe aleatoire.
    "life_3d_bays": RuleSet.from_spec("B6/S5,6,7"),
    # Life 4555 (Bays 1987) : la premiere regle 3D decouverte, et selon Bays
    # l'une des deux seules "dignes du nom" avec la 5766. Plage de survie plus
    # etroite (proportionnellement proche du B3/S23 2D) : la soupe reste
    # dynamique beaucoup plus longtemps, avec des gliders connus.
    "life_3d_4555": RuleSet.from_spec("B5/S4,5"),
    "life_3d_5655": RuleSet.from_spec("B5/S5,6"),
    "life_3d_6855": RuleSet.from_spec("B5/S6,7,8"),
    "life_3d_highlife": RuleSet.from_spec("B5,6,7/S6,7,8"),
    "life_4d_balanced": RuleSet.from_spec("B18,19,20/S18,19,20,21"),
}


def get_rule(name_or_spec: str) -> RuleSet:
    if name_or_spec in PRESET_RULES:
        return PRESET_RULES[name_or_spec]
    return RuleSet.from_spec(name_or_spec)
