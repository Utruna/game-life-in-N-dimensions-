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
    # ATTENTION : mesuree eteinte en DEUX generations (33 081 -> 163 -> 0), et
    # a toutes les densites initiales de 10 % a 40 %. Ce n'est pas le seuil qui
    # la tue mais la largeur : sa fenetre de survie ne fait que 4 de large alors
    # que le nombre de voisins vivants se disperse d'environ +/-4 autour de la
    # moyenne, donc elle laisse echapper la majorite des cellules quel que soit
    # le reglage. Conservee pour memoire, elle ne produit rien de visible.
    "life_4d_balanced": RuleSet.from_spec("B18,19,20/S18,19,20,21"),
    # Regles 4D retenues apres balayage systematique (20^4, 3 a 6 seeds,
    # 200 a 300 generations, bords non toroidaux). Deux mesures les separent :
    # le remplissage a l'equilibre, et surtout le RENOUVELLEMENT, la part de
    # cellules qui changent d'etat a chaque pas. C'est lui qui distingue une
    # structure d'un scintillement, et il donne un reperage utile : la regle 3D
    # que nous utilisons (B5,6,7/S6,7,8) tourne a 136 %.
    #
    # Le levier qui commande le renouvellement n'est pas le seuil mais la
    # LARGEUR de la fenetre de survie. Avec 80 voisins, le nombre de voisins
    # vivants se disperse d'environ +/-4 autour de la moyenne : une fenetre de
    # 4 laisse echapper la majorite des cellules a chaque pas (d'ou la mort de
    # life_4d_balanced), une fenetre de 6 les retient et les motifs persistent.
    #
    # Agitee : 20 % de remplissage, 142 % de renouvellement. C'est l'equivalent
    # 4D exact de notre HighLife 3D, pas une regle plus chaotique qu'elle.
    "life_4d_chaos": RuleSet.from_spec("B12,13,14/S12,13,14,15"),
    # Calme : 26 % de remplissage pour seulement 80 % de renouvellement, soit
    # 40 % moins agitee que les deux precedentes. Les structures tiennent
    # plusieurs generations au lieu de se renouveler en bloc. Stable sur
    # 6 seeds (derive de 1 % entre les generations 150 et 300).
    "life_4d_calm": RuleSet.from_spec("B10,11,12/S12,13,14,15,16,17"),
    # Intermediaire, un peu moins dense que calm pour une agitation voisine.
    "life_4d_mid": RuleSet.from_spec("B11,12/S11,12,13,14,15,16"),
}


def get_rule(name_or_spec: str) -> RuleSet:
    if name_or_spec in PRESET_RULES:
        return PRESET_RULES[name_or_spec]
    return RuleSet.from_spec(name_or_spec)
