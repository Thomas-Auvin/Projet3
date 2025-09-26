### Fonction pour le nettoyage du dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Fonction qui est utile pour définir un booléen pour savoir si le batiment a plusieurs usages ou non.

def Multi_usage(row :str) -> str:
    if pd.isna(row["LargestPropertyUseType"]) and pd.isna(row["SecondLargestPropertyUseType"]) and pd.isna(row["ThirdLargestPropertyUseType"]):
        return "NaN"
    elif pd.notna(row["LargestPropertyUseType"]) and pd.isna(row["SecondLargestPropertyUseType"]) and pd.isna(row["ThirdLargestPropertyUseType"]):
        return "Mono usage"
    else:
        return "Multi usage"


#Fonction de définition de groupe des années de construction. 
#Pour définir cette variable j'ai demandé à chatgpt de mon trouver la legislation américaine sur le norme de construction. Il m'a trouvé trois grands types 
# 	1) Avant 1975	et la première norme énergétique (ASHRAE 90 en 1975) qui sera donc la catégorie "1900-1974"
#   2) Entre ASHRAE 90 (1975) et création de l’IECC (2000) qui sera donc la catégorie "1975–1999"
#   3) Et enfin l'apparition de la norme IECC. Cette norme est découpé en plusieurs partie mais pour éviter de trop petits groupe elle a été regroupé en une seule 
#       qui sera donc la catégorie "IECC 2000–2020"
# Par sécurité pour de potentiels futurs ajout la catégorie "IECC 2020+" a été crée pour stocker d'autres batiments qui apparaitrait dans un futur dataset

def Categorie_anne_construction(year : int) -> str:
    if year < 1975:
        return "1900-1974"
    elif year < 2000:
        return "1975–1999"
    elif year < 2020:
        return "IECC 2000–2020"
    else:
        return "IECC 2020+"

# fonction de regroupement par nombre d'étages pour créer des groupes cohérents en fonction du nombre d'étages

def Categorie_nb_etage(etage : int) -> str:
    if etage < 5:
        return "Bas"
    elif etage < 10:
        return "Moyen"
    else:
        return "Haut"


#fonction pour définir si un batiment est un ensemble de batiment ou un batiment unique. 

def Categorie_nb_batiment(batiment: int) -> str:
    if batiment < 2:
        return "Bat_unique"
    else:
        return "Bat_multiple"


### Deux fonctions qui fonctionnent ensemble pour définir la part de l'utilisation des batiments réservés à l'habitation et la part réserv à la non habitation
### Cela servira ensuite à ajuster la target pour tenter d'enlever la part que représente une habitation dans la consommation. 

# 1) Quels usages comptent comme "habitation" ?
HAB_TYPES = {"multifamily housing"}  # en minuscules

# 2) Les paires (usage, surface) EXACTES du fichier
PAIRS = [
    ("LargestPropertyUseType", "LargestPropertyUseTypeGFA"),
    ("SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA"),
    ("ThirdLargestPropertyUseType", "ThirdLargestPropertyUseTypeGFA"),
]

def est_habitation(usage):
    """Retourne True si l'usage est une habitation (test insensible à la casse)."""
    if pd.isna(usage):
        return False
    return str(usage).strip().lower() in HAB_TYPES

def calcul_part_habitation(row) -> float:
    surface_totale = 0.0
    surface_habitation = 0.0

    for col_usage, col_surface in PAIRS:
        usage = row.get(col_usage)
        surface = pd.to_numeric(row.get(col_surface), errors="coerce")
        if pd.isna(surface) or surface <= 0:
            continue

        surface_totale += float(surface)
        if est_habitation(usage):
            surface_habitation += float(surface)

    if surface_totale == 0:
        return pd.Series([np.nan, np.nan])  # pas de division par zéro

    part_hab = round(100 * surface_habitation / surface_totale, 2)
    part_non_hab = round(100 - part_hab, 2)
    return pd.Series([part_hab, part_non_hab])
