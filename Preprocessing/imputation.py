import pandas as pd
import numpy as np

def get_imputation_maps(df, target_col):
    '''
    Diese Funktion erstellt ein Mapping für die Imputation von fehlenden Werten.
    1. Schaue im Kontext groupings[0], ob du eindeutig die Werte zuweisen kannst.

    2. Falls nicht möglich (keine eindeutigen gefunden)
    -> schaue im Kontext groupings[1] und bilde den Mean 

    3. Falls nicht möglich (keine Zeilen im Kontext groupings[1])
    -> schau im Kontext grouping[2] und bilde den Mean    
    '''
    groupings = [
        ['brand', 'model', 'year', 'power_ps', 'fuel_type', 'transmission_type'],
        ['brand', 'model', 'power_ps', 'fuel_type', 'transmission_type'],
        ['model', 'fuel_type']
    ]
    maps = []

    group_cols = groupings[0]
    counts = df.groupby(group_cols)[target_col].nunique().reset_index(name='unique_count')
    unique_contexts = counts.query('unique_count == 1').drop(columns='unique_count')
    map_unique = (
        df.dropna(subset=[target_col])
          .merge(unique_contexts, on=group_cols, how='inner')
          .groupby(group_cols)[target_col]
          .first()
    )
    maps.append((group_cols, map_unique))

    for group_cols in groupings[1:]:
        map_mean = (
            df.dropna(subset=[target_col])
              .groupby(group_cols)[target_col]
              .mean()
              .round(2)
        )
        maps.append((group_cols, map_mean))

    return maps


def apply_imputation(df, target_col, maps):
    ''' 
    Füllt fehlende Werte in target_col anhand der übergebenen Imputations-Mappings (maps).
    Geht dabei stufenweise vor zuerst spezifische, dann allgemeinere Zuordnung
    '''
    df = df.copy()
    for group_cols, impute_map in maps:
        def fill(row):
            if pd.isna(row[target_col]):
                key = tuple(row.get(col) for col in group_cols)
                return impute_map.get(key, np.nan)
            return row[target_col]
        df[target_col] = df.apply(fill, axis=1)

    return df

from sklearn.base import BaseEstimator, TransformerMixin

class ContextImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.maps_ = None

    def fit(self, X, y=None):
        # X ist DataFrame
        self.maps_ = get_imputation_maps(X.copy(), target_col=self.target_col)
        return self
    
    def transform(self, X):
        return apply_imputation(X.copy(), target_col=self.target_col, maps=self.maps_)