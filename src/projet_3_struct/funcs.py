### Fonction pour le nettoyage du dataframe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def Multi_usage(row):
    if pd.isna(row["LargestPropertyUseType"]) and pd.isna(row["SecondLargestPropertyUseType"]) and pd.isna(row["ThirdLargestPropertyUseType"]):
        return "NaN"
    elif pd.notna(row["LargestPropertyUseType"]) and pd.isna(row["SecondLargestPropertyUseType"]) and pd.isna(row["ThirdLargestPropertyUseType"]):
        return "Mono usage"
    else:
        return "Multi usage"
    
def Categorie_anne_construction(year):
    if year < 1975:
        return "1900-1974"
    elif year < 2000:
        return "1975–1999"
    elif year < 2020:
        return "IECC 2000–2018"
    else:
        return "IECC 2020+"

def Categorie_nb_etage(etage):
    if etage < 5:
        return "Bas"
    elif etage < 10:
        return "Moyen"
    else:
        return "Haut"
    
def Categorie_nb_batiment(batiment):
    if batiment < 2:
        return "Bat_unique"
    else:
        return "Bat_multiple"

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

def calcul_part_habitation(row):
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
