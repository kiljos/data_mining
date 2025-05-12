import pandas as pd
import numpy as np
import re
import gc  # Garbage Collector zur Speicherverwaltung


def drop_records_brand_equal_model(df):
    """
    Entfernt alle Zeilen, in denen brand und model nach Normalisierung identisch sind.
    """
    def normalize(text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^a-z0-9]", "", text.lower())

    df = df.copy()
    df["brand_norm"] = df["brand"].apply(normalize)
    df["model_norm"] = df["model"].apply(normalize)

    # Zeilen behalten, bei denen die Normalisierungen unterschiedlich sind
    df = df[df["brand_norm"] != df["model_norm"]].reset_index(drop=True)

    df.drop(columns=["brand_norm", "model_norm"], inplace=True)
    return df


def clean_fuel_consumption(value):
    """
    Extrahiert den Verbrauch in l/100 km als float.
    Alle anderen Einheiten und ungültige Werte werden zu NaN.
    """
    if pd.isna(value) or 'l/100 km' not in str(value):
        return np.nan
    try:
        # Zahl vor dem Leerzeichen, Komma zu Punkt
        return float(str(value).split(' ')[0].replace(',', '.'))
    except:
        return np.nan


def preprocessing_pipeline(path='../data.csv'):
    # Daten laden
    df = pd.read_csv(path)

    # Duplikate entfernen und unerwünschte Index-Spalte droppen
    df = df.drop_duplicates(subset=[
        'brand', 'model', 'color', 'registration_date', 'year',
        'price_in_euro', 'power_kw', 'power_ps', 'transmission_type',
        'fuel_type', 'fuel_consumption_l_100km', 'fuel_consumption_g_km',
        'mileage_in_km', 'offer_description'
    ])
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Nur relevante fuel types
    df = df[df['fuel_type'].isin(['Diesel', 'Petrol', 'Hybrid', 'LPG', 'Electric'])].reset_index(drop=True)

    # Jahresfilter anwenden
    valid_years = list(df['year'].unique()[:29])
    df = df[df['year'].isin(valid_years)]

    # Fuel consumption in l/100 km bereinigen
    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_fuel_consumption)

    # Ausreißer für Verbrenner auf NaN setzen
    mask_outliers = (df['fuel_consumption_l_100km'] > 23) & df['fuel_type'].isin(['Diesel', 'Petrol', 'Hybrid', 'LPG'])
    df.loc[mask_outliers, 'fuel_consumption_l_100km'] = np.nan

    # Drop brand==model
    df = drop_records_brand_equal_model(df)

    # E-Autos: Verbrauch auf 0 setzen, Reichweite extrahieren
    e_mask = df['fuel_type'] == 'Electric'
    # Setze Verbrauch l/100 km auf 0
    df.loc[e_mask, 'fuel_consumption_l_100km'] = 0.0
    # Extrahiere Reichweite aus fuel_consumption_g_km
    range_mask = e_mask & df['fuel_consumption_g_km'].astype(str).str.contains(r'km Reichweite', na=False)
    df.loc[range_mask, 'electric_range'] = (
        df.loc[range_mask, 'fuel_consumption_g_km']
          .astype(str)
          .str.extract(r'(\d+)', expand=False)
          .astype(float)
    )
    # Nicht-Elektro auf 0
    df.loc[~e_mask, 'electric_range'] = 0.0

    # Numerische Typen setzen
    for col in ['power_ps', 'power_kw']:
        df[col] = df[col].astype(float)
    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Datum in numerische Merkmale umwandeln
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month

    # Nicht mehr benötigte Spalten entfernen
    df.drop(columns=[
        'registration_date', 'power_kw', 'offer_description', 'fuel_consumption_g_km'
    ], inplace=True)

    # Speicher freigeben
    gc.collect()

    return df