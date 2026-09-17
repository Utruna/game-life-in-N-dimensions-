from __future__ import annotations

from engine.rules import RuleSet

# Presets 3D issus de la littérature sur le Life 3D (notamment Carter Bays).
# Les règles exactes dépendent de la variante; ces presets servent de base d'exploration.
PRESET_RULES: dict[str, RuleSet] = {
    "life_3d_bays": RuleSet.from_spec("B6/S5,6,7"),
    "life_3d_highlife": RuleSet.from_spec("B5,6,7/S6,7,8"),
    "life_4d_balanced": RuleSet.from_spec("B18,19,20/S18,19,20,21"),
}


def get_rule(name_or_spec: str) -> RuleSet:
    if name_or_spec in PRESET_RULES:
        return PRESET_RULES[name_or_spec]
    return RuleSet.from_spec(name_or_spec)
