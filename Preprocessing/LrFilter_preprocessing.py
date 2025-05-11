import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def preprocessing_pipeline_lrfilter(df):
        
        
  

    
    

    # Abtrennen der Textbausteine + aussortieren von Zeilen, die verrutscht sind, in der beiden fuel consumption Spalte 
                
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
                        
    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_fuel_consumption)
     


    
    # Fixe wo model = brand, versuche eindeutig Model zuzuweisen sonst droppen 
    df = fix_model_brand_conflicts(df) 

    
    
    


    # Spalten ins numerische umwandeln
    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['power_ps'] = pd.to_numeric(df['power_ps'], errors='coerce')
    
    
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    df["age"] = 2023 - df["year"]
    

    # Droppe alle Zeilen, in denen null values vorkommen
    df = df.dropna()

    # Outlier Detection für fuel
    q1_fuel= df['fuel_consumption_l_100km'].quantile(0.25)
    q3_fuel=df['fuel_consumption_l_100km'].quantile(0.75)
    iqr_fuel = q3_fuel - q1_fuel    
    lower_bound_fuel = q1_fuel - 1.5 * iqr_fuel
    upper_bound_fuel = 25 #fix alles löschen was über 25 l/100km ist
    df = df[(df['fuel_consumption_l_100km'] >= lower_bound_fuel) & (df['fuel_consumption_l_100km'] <= upper_bound_fuel)]
    
    # Funktion Outlier detection für Preis & Mileage 
    def detect_outliers_iqr(df, group_col, target_col):
        outlier_flags = pd.Series(False, index=df.index)

        for name, group in df.groupby(group_col):
            if len(group) < 2:
                continue

            q1 = group[target_col].quantile(0.25)
            q3 = group[target_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (group[target_col] < lower) | (group[target_col] > upper)
            outlier_flags.loc[group[mask].index] = True

        return outlier_flags

    df.drop(columns=['fuel_consumption_g_km','power_kw',"registration_date","year"], inplace=True) 
    
    df['outlier_model_mileage'] = detect_outliers_iqr(df, ['brand', 'model'], 'mileage_in_km')

    # Entferne alle Rows, die bei Preis & Mileage ein Outlier sind
    df = df[(~df['outlier_model_mileage'])].copy()
    
    # Lösche die Outlier-Spalte, da sie nicht mehr benötigt wird
    df.drop(columns=['outlier_model_mileage'], inplace=True)

    df.drop(columns=['offer_description'], inplace=True)

    
    return df
