### Fonction pour le nettoyage du dataframe

import pandas as pd
import numpy as np
from typing import Iterable


# fonction de réassignation des catégories pour s'assurer d'un bon encodage

def reassignation_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cast = {
        "DataYear": "category",
        "OSEBuildingID": "string",
        "ZipCode": "string",
        "CouncilDistrictCode": "string",
        "Comments": "string",
        "DefaultData": "string",
    }
    for c, t in cast.items():
        if c in out.columns:
            out[c] = out[c].astype(t)
    return out


# fonction pour retirer les colonnes jugées inutiles 

def retirer_colonnes_inutiles(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Comments","DataYear","TaxParcelIdentificationNumber","City","State"], errors="ignore")

# fonction pour normaliser les écritures des catégories repérés comme mal écrites 
# (normalisation_ecriture pour la features Neighborhood)
# (normalisation_buildingtype pour la features BuildingType)

def normalisation_ecriture(df: pd.DataFrame) -> pd.DataFrame:
    if "Neighborhood" not in df.columns:
        return df
    out = df.copy()
    out["Neighborhood"] = (
        out["Neighborhood"].astype("string").str.strip().str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\bNORTH WEST\b", "NORTHWEST", regex=True)
        .replace({"DELRIDGE NEIGHBORHOODS": "DELRIDGE"})
    )
    return out

def normalisation_buildingtype(df):
    if "BuildingType" not in df.columns:
        return df
    out = df.copy()
    out["BuildingType"] = out["BuildingType"].astype("string").replace(
        to_replace=r"^Nonresidential\s+(COS|WA)$",
        value="NonResidential",
        regex=True,
    )
    return out

# fonction de réassignation d'un NA pour les données repérés comme aberrante (ici une seule)
def reassignation_outlier(df):
    if "NumberofFloors" not in df.columns:
        return df
    out = df.copy()
    out.loc[out["NumberofFloors"] == 99, "NumberofFloors"] = np.nan
    return out

#fonction pour ajouter la colonne usage multiple

def ajout_usage_multiple(df, usage_func):
    out = df.copy()
    out["Usage_multiple"] = out.apply(usage_func, axis=1)
    return out

# fonction pour filtrer et enlever les batiments qui n'aurait qu'une utilisation d'habitation

def drop_mono_family_et_campus(df):
    out = df.copy()
    if {"BuildingType","Usage_multiple"}.issubset(out.columns):
        non_valid = {'Multifamily LR (1-4)','Multifamily MR (5-9)', 'Multifamily HR (10+)'}
        cond_supp = out["BuildingType"].isin(non_valid) & (out["Usage_multiple"] == "Mono usage")
        out = out.loc[~cond_supp]
    if "BuildingType" in out.columns:
        out = out.loc[~(out["BuildingType"] == "Campus")]
    return out

# fonction pour filter les batiments ou la cible est à 0

def filtre_energy_gt0(df):
    if "SiteEnergyUseWN(kBtu)" not in df.columns:
        return df
    return df.loc[df["SiteEnergyUseWN(kBtu)"] > 0]

# fonction pour filter les batiments ou la cible seconde est à 0

def filtre_co2_gt0(df):
    if "TotalGHGEmissions" not in df.columns:
        return df
    return df.loc[df["TotalGHGEmissions"] > 0]



### fonction pour calculer la part d'habitation pour les batiments qui aurait un usage multiple. 
### Cette fonction est aussi utile pour repérer des batiments qui n'aurait pas un usage autre 
### qu'habitation et qui aurait été mal classé. La fonction est en deux parties avec la définition de la part
### puis le filtrage de celle qui aurait uniquement une part habitation. 


# Par défaut, on considère "multifamily housing" comme de l'habitation.
DEFAULT_HAB_TYPES = {"multifamily housing"}

# Paires (usage, surface)
DEFAULT_PAIRS = [
    ("LargestPropertyUseType", "LargestPropertyUseTypeGFA"),
    ("SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA"),
    ("ThirdLargestPropertyUseType", "ThirdLargestPropertyUseTypeGFA"),
]

def add_part_habitation(
    df: pd.DataFrame,
    hab_types: Iterable[str] = None,
    pairs: list[tuple[str, str]] = None,
    out_cols: tuple[str, str] = ("PartHabitation(%)", "PartNonHabitation(%)")
) -> pd.DataFrame:
    """
    Calcule les parts d'habitation (%) et non-habitation (%) à partir des colonnes d'usages et de surfaces GFA.
    Renvoie un DF avec deux nouvelles colonnes out_cols.
    """
    if hab_types is None:
        hab_types = DEFAULT_HAB_TYPES
    if pairs is None:
        pairs = DEFAULT_PAIRS

    out = df.copy()

    # Filtrer aux colonnes réellement présentes
    pairs_presentes = [(u, g) for (u, g) in pairs if (u in out.columns and g in out.columns)]
    if not pairs_presentes:
        # Rien à faire si aucune paire n'existe
        out[out_cols[0]] = np.nan
        out[out_cols[1]] = np.nan
        return out

    usage_cols = [u for (u, _) in pairs_presentes]
    gfa_cols   = [g for (_, g) in pairs_presentes]

    # GFA numériques (négatifs -> 0)
    gfa = out[gfa_cols].apply(pd.to_numeric, errors="coerce")
    gfa_pos = gfa.where(gfa > 0, 0.0)

    # Total des surfaces
    surface_totale = gfa_pos.sum(axis=1)

    # Masques habitation (insensibles à la casse)
    sum_hab = 0.0
    for u_col, g_col in pairs_presentes:
        is_hab = (
            out[u_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(set(s.lower() for s in hab_types))
        )
        sum_hab = sum_hab + gfa_pos[g_col].where(is_hab, 0.0)

    # Pour éviter division par 0
    with np.errstate(divide="ignore", invalid="ignore"):
        part_hab = (sum_hab / surface_totale) * 100

    part_hab = part_hab.round(2)
    part_non = (100 - part_hab).round(2)

    # Si surface_totale == 0 → NaN
    zero_mask = surface_totale.eq(0)
    part_hab = part_hab.mask(zero_mask, np.nan)
    part_non = part_non.mask(zero_mask, np.nan)

    out[out_cols[0]] = part_hab
    out[out_cols[1]] = part_non
    return out


def filter_non_hab_gt0(df: pd.DataFrame, col: str = "PartNonHabitation(%)") -> pd.DataFrame:
    """Garde uniquement les lignes où la part non-habitation est > 0%."""
    if col not in df.columns:
        return df
    return df.loc[df[col] > 0].copy()

###fonction utile pour compter l'effet du nettoyage.

def log_counts(df, tag):
    print(f"\n[{tag}] rows={len(df)}")
    print(df["BuildingType"].value_counts(dropna=False))
    return df