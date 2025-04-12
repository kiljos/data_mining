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
import re
import gc  # Garbage Collector zur Speicherverwaltung



def fix_model_brand_conflicts(df):
    '''Diese Funktion überprüft, ob es Zeilen gibt, in denen brand = model ist. In diesen Zeilen haben wir keine Informationen
    über das Model. 
    
    Um die Zeilen aber nicht direkt zu droppen, wird vorher geschaut, ob man über bestimmte Spalten das Model eindeutig zuornden kann.
    Ist eine eindeutige Zuordnung möglich, dann überschreiben wir die ursprüngliche Ausprägung in model. Ist keine Zuordnung möglich, dann wird die Zeile
    gedroppt.
    '''

    def normalize(text):
        if pd.isna(text):
            return ""
        return re.sub(r'[^a-z0-9]', '', text.lower())

    df['brand_norm'] = df['brand'].apply(normalize)
    df['model_norm'] = df['model'].apply(normalize)

    mask_same = df['brand_norm'] == df['model_norm']

    problem_rows = df[mask_same].copy()
    clean_rows = df[~mask_same].copy()

    grouped_models = clean_rows \
        .groupby(['brand', 'power_ps', 'fuel_consumption_g_km', 'transmission_type', 'fuel_type'])['model'] \
        .unique().reset_index() 

    grouped_models = grouped_models[grouped_models['model'].apply(len) == 1] # nur kontexte bei denen model unique ist (ein element in der liste)
    grouped_models['model'] = grouped_models['model'].apply(lambda x: x[0]) # nimm nur das erste element aus der liste

    problem_fixed = problem_rows.merge(grouped_models, on=['brand', 'power_ps', 'fuel_consumption_g_km', 'transmission_type', 'fuel_type'],
                                       how='left', suffixes=('', '_fixed'))

    recovered = problem_fixed[problem_fixed['model_fixed'].notna()].copy()
    recovered['model'] = recovered['model_fixed']
    recovered = recovered.drop(columns=['model_fixed'])

    final_df = pd.concat([clean_rows, recovered], ignore_index=True) \
                 .drop(columns=['brand_norm', 'model_norm'])

    return final_df


def preprocessing_pipeline():
        
    # Daten laden
    df = pd.read_csv('data.csv')

    # Entferne unnötige Spalten
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'offer_description' in df.columns:
        df = df.drop('offer_description', axis=1)

    # Entferne die Duplikate
    df = df.drop_duplicates()

    # Filtere die Jahr-Spalte (Achtung: Dieser Filter nimmt die ersten 29 eindeutigen Werte; evtl. anpassen)
    # wird gemacht um auf die Verschiebung von den Werten zu reagieren
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
    def clean_fuel_consumption(value): # Bei Elektroautos steht Reichweite
        if pd.isna(value) or 'l/100 km' not in str(value):
            return np.nan
        try:
            return float(value.split(' ')[0].replace(',', '.'))
        except:
            return np.nan

    def clean_fuel_consumption_g(value):
        if pd.isna(value) or value == '- (g/km)' or 'g/km' not in str(value): # Bei Elektroautos steht Reichweite
            return np.nan
        try:
            return float(value.split(' ')[0])
        except:
            return np.nan

    # Säubere Kraftstoffverbrauchsspalten
    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_fuel_consumption)
    df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_fuel_consumption_g)

    # Konvertiere die Powerspalten
    for col in ['power_ps', 'power_kw']:
        df[col] = df[col].astype(float)

    # Konvertiere registration_date in datetime und extrahiere Monat und Jahr als Features
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month
    df['registration_year'] = df['registration_date'].dt.year
    df = df.drop('registration_date', axis=1)

    # Numerische Konvertierung von mileage und price
    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')

    # Fixe wo model = brand, versuche eindeutig Model zuzuweisen sonst droppen 
    df = fix_model_brand_conflicts(df)

    # Model war hier in einer Zeile falsch geschrieben
    df['model'] = df['model'].replace('Saab 93', 'Saab 9-3')

    # drop power_kw da Vielfaches von power_ps
    if 'power_kw' in df.columns:
        df = df.drop('power_kw', axis=1)
    
    # Reduziert R2
    if 'year' in df.columns:
        df = df.drop('year', axis=1)

    # Entferne Zeilen mit fehlenden Werten
    print(f"Dataset-Größe vor dem Entfernen von NaN-Werten: {len(df)}")
    df = df.dropna()
    print(f"Dataset-Größe nach dem Entfernen von NaN-Werten: {len(df)}")


    def filter_models_with_min_count(df, model_col='model', min_count=100):
        '''Diese Funktion filtert den DF auf die Modelle ein die in einer bestimmten
        Frequenz vorkommen'''
        model_counts = df[model_col].value_counts()
        models_over_threshold = model_counts[model_counts >= min_count].index
        df_filtered = df[df[model_col].isin(models_over_threshold)]
        return df_filtered
    
    # Nur die Modelle, die mind. 100 mal vorkommen
    df = filter_models_with_min_count(df, 'model', 100)

    # Definiere das Ziel
    y = df['price_in_euro']
    X = df.drop(['price_in_euro'], axis=1)
    
    print(X.columns)

    print(X.info())

    # Identifiziere numerische und kategoriale Spalten
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # (Hinweis: Der Limiting-Block für kategoriale Features wurde entfernt.)

    # Speicher freigeben
    gc.collect()

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['model']) #stratify auf basis von model spalte
    return X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features
