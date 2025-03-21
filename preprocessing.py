
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gc  # Garbage Collector zur Speicherverwaltung


def preprocessing_pipeline():
        
    # Daten laden
    df = pd.read_csv('data.csv')

    # Entferne unnötige Spalten
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'offer_description' in df.columns:
        df = df.drop('offer_description', axis=1)

    # Filtere die Jahr-Spalte (Achtung: Dieser Filter nimmt die ersten 29 eindeutigen Werte; evtl. anpassen)
    print('Werte in der year-Spalte vor dem Filtern:')
    print(df['year'].unique(), '\n')
    yearsToFilter = list(df['year'].unique()[:29])
    filt = [val in yearsToFilter for val in df['year']]
    df = df[filt]
    print('Werte nach dem Filtern:')
    print(df['year'].unique())

    # Stelle sicher, dass year numerisch ist
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Funktionen zur Bereinigung der Kraftstoffverbrauchswerte
    def clean_fuel_consumption(value):
        if pd.isna(value) or value == '- (g/km)':
            return np.nan
        try:
            return float(value.split(' ')[0].replace(',', '.'))
        except:
            return np.nan

    def clean_fuel_consumption_g(value):
        if pd.isna(value) or value == '- (g/km)':
            return np.nan
        try:
            return float(value.split(' ')[0])
        except:
            return np.nan

    # Säubere Kraftstoffverbrauchsspalten
    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_fuel_consumption)
    df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_fuel_consumption_g)

    # Konvertiere registration_date in datetime und extrahiere Monat und Jahr als Features
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month
    df['registration_year'] = df['registration_date'].dt.year
    df = df.drop('registration_date', axis=1)

    # Numerische Konvertierung von mileage und price
    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')

    # Entferne Zeilen mit fehlenden Werten
    print(f"Dataset-Größe vor dem Entfernen von NaN-Werten: {len(df)}")
    df = df.dropna()
    print(f"Dataset-Größe nach dem Entfernen von NaN-Werten: {len(df)}")

    # Definiere das Ziel
    y = df['price_in_euro']
    X = df.drop(['price_in_euro'], axis=1)

    # Identifiziere numerische und kategoriale Spalten
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # (Hinweis: Der Limiting-Block für kategoriale Features wurde entfernt.)

    # Speicher freigeben
    gc.collect()

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test , categorical_features , numeric_features