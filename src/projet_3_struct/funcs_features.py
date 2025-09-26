# projet_3_struct/funcs_features.py
"""
Fonctions de features

- Création des groupes (année de construction, nb étages, nb bâtiments) + labels (codes)
- Flags 0/1 : gaz/élec/steam mesurés
- log1p sur PropertyGFATotal / PropertyGFAParking
- One-hot (get_dummies) sur les colonnes catégorielles choisies
- Sélection stricte des colonnes effectivement utilisées pour l'entraînement
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Optional

# Fonctions de catégorisation
from projet_3_struct.funcs import (
    Categorie_anne_construction,
    Categorie_nb_etage,
    Categorie_nb_batiment,
)

# Ordres utilisés pour les catégories
ORDRE_ANNEE  = ['1900-1974', '1975–1999', 'IECC 2000–2018']
ORDRE_ETAGES = ['Bas', 'Moyen', 'Haut']
ORDRE_BATS   = ['Bat_unique', 'Bat_multiple']

# Colonnes par défaut
DEFAULT_LOG_COLS = ['PropertyGFATotal', 'PropertyGFAParking']
DEFAULT_CAT_COLS = ['BuildingType', 'CouncilDistrictCode', 'Usage_multiple',
                    'PrimaryPropertyType', 'Neighborhood']

def _flag_measured(s: pd.Series) -> pd.Series:
    """Retourne 0/1 si la mesure est non nulle (NaN et 0 -> 0)."""
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    return (x != 0).astype("int8")

def build_manual_features(
    dfin: pd.DataFrame,
    *,
    include_defaultdata: bool = False,
    log_cols: Optional[Sequence[str]] = None,
    cat_cols: Optional[Sequence[str]] = None,
    drop_text_groups: bool = True,
    include_measure_flags: bool = True,   # <— nouveau param (True par défaut)
) -> pd.DataFrame:
    """
    Paramètres
    ----------
    dfin : DataFrame d'entrée (nettoyé)
    log_cols : colonnes numériques à passer en log1p (défaut: PropertyGFATotal/PropertyGFAParking)
    cat_cols : colonnes catégorielles à one-hot encoder (défaut: DEFAULT_CAT_COLS)
    drop_text_groups : supprime les colonnes texte des groupes (on garde les *_label)
    include_measure_flags : ajoute 3 indicateurs 0/1 (gaz/élec/steam mesurés)

    Retour
    ------
    DataFrame enrichi (groupes+labels, flags, logs, dummies)
    """
    d = dfin.copy()

    # Groupes & labels
    if 'YearBuilt' in d.columns:
        d['Groupe_anne_construction'] = d['YearBuilt'].apply(Categorie_anne_construction)
        d['Groupe_anne_construction'] = pd.Categorical(
            d['Groupe_anne_construction'], categories=ORDRE_ANNEE, ordered=True
        )
        d['Groupe_anne_construction_label'] = d['Groupe_anne_construction'].cat.codes

    if 'NumberofFloors' in d.columns:
        d['Groupe_nb_etages'] = d['NumberofFloors'].apply(Categorie_nb_etage)
        d['Groupe_nb_etages'] = pd.Categorical(
            d['Groupe_nb_etages'], categories=ORDRE_ETAGES, ordered=True
        )
        d['Groupe_nb_etages_label'] = d['Groupe_nb_etages'].cat.codes

    if 'NumberofBuildings' in d.columns:
        d['Groupe_nb_batiments'] = d['NumberofBuildings'].apply(Categorie_nb_batiment)
        d['Groupe_nb_batiments'] = pd.Categorical(
            d['Groupe_nb_batiments'], categories=ORDRE_BATS, ordered=True
        )
        d['Groupe_nb_batiments_label'] = d['Groupe_nb_batiments'].cat.codes

    # Flags 0/1 : présence de mesures non nulles (robuste aux NaN)
    if include_measure_flags:
        if 'NaturalGas(therms)' in d.columns:
            d['Conso_gaz_mesure'] = _flag_measured(d['NaturalGas(therms)'])
        if 'SteamUse(kBtu)' in d.columns:
            d['Emission_steam_mesure'] = _flag_measured(d['SteamUse(kBtu)'])

    # DefaultData : 0/1 uniquement si explicitement demandé
    if include_defaultdata and 'DefaultData' in d.columns:
        mapper = {True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0, 1: 1, 0: 0}
        d['DefaultData'] = pd.Series(d['DefaultData']).map(mapper).fillna(0).astype('int64')

    # log1p sur colonnes numériques
    for c in (log_cols or DEFAULT_LOG_COLS):
        if c in d.columns:
            d[c] = np.log1p(pd.to_numeric(d[c], errors='coerce').clip(lower=0))

    # One-hot sur les catégorielles choisies
    cats = [c for c in (cat_cols or DEFAULT_CAT_COLS) if c in d.columns]
    if cats:
        dummies = pd.get_dummies(d[cats], drop_first=False)
        d = pd.concat([d.drop(columns=cats), dummies], axis=1)

    # Retire les colonnes texte des groupes (on garde leurs labels)
    if drop_text_groups:
        to_drop = [c for c in ['Groupe_anne_construction', 'Groupe_nb_etages', 'Groupe_nb_batiments']
                   if c in d.columns]
        d = d.drop(columns=to_drop, errors='ignore')

    return d

# === Sélection stricte des colonnes utilisées pour l'entraînement ===
KEEP_BASE_NUM = [
    'PropertyGFATotal',
    'PropertyGFAParking',
    'Groupe_anne_construction_label',
    'Groupe_nb_etages_label',
    'Groupe_nb_batiments_label',
    'Latitude',
    'Longitude',
    # Flags mesurés :
    'Conso_gaz_mesure',
    'Emission_steam_mesure',
]

# Dummies conservées (préfixes)
DUMMY_PREFIXES = [
    'BuildingType_',
    'CouncilDistrictCode_',
    'Usage_multiple_',
    'PrimaryPropertyType_',
    'Neighborhood_',   # <- préfixe plus précis
    'DefaultData_',
]

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve uniquement :
      - les colonnes numériques "de base" (KEEP_BASE_NUM) si présentes
      - toutes les colonnes dummies dont le nom commence par un des DUMMY_PREFIXES
    """
    cols: list[str] = [c for c in KEEP_BASE_NUM if c in df.columns]
    cols += [c for c in df.columns if any(c.startswith(pfx) for pfx in DUMMY_PREFIXES)]
    return df[cols].copy()
