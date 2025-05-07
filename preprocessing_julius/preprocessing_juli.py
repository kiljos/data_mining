import pandas as pd
import numpy as np 
import re

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

    df.loc[:, 'brand_norm'] = df['brand'].apply(normalize)
    df.loc[:, 'model_norm'] = df['model'].apply(normalize)

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

def calculate_fuel_consumption(row):

        conversion_factor = 0.043103448275862

        if pd.isna(row['fuel_consumption_l_100km']) or row['fuel_consumption_l_100km'] == 0:
            if pd.notna(row['fuel_consumption_g_km']) and row['fuel_consumption_g_km'] != 0:
                return row['fuel_consumption_g_km'] * conversion_factor
            else:
                return np.nan
        else:
            return row['fuel_consumption_l_100km']

def impute_fuel_staged(df, target_col):
    groupings = [
        ['brand', 'model', 'year', 'power_ps', 'fuel_type', 'transmission_type'],  # Stufe 1: eindeutig
        ['brand', 'model', 'power_ps', 'fuel_type', 'transmission_type'],          # Stufe 2: Mittelwert
        ['model', 'fuel_type']                                                     # Stufe 3: fallback
    ]
    
    # Stufe 1: nur eindeutige Fälle (nunique == 1)
    group_cols = groupings[0]
    counts = df.groupby(group_cols)[target_col].nunique().reset_index(name='unique_count')
    unique_contexts = counts.query('unique_count == 1').drop(columns='unique_count')

    fuel_map_unique = (
        df.dropna(subset=[target_col])
          .merge(unique_contexts, on=group_cols, how='inner')
          .groupby(group_cols)[target_col]
          .first()
    )

    def fill_unique(row):
        if pd.isna(row[target_col]):
            key = tuple(row[col] for col in group_cols)
            return fuel_map_unique.get(key, np.nan)
        return row[target_col]

    df[target_col] = df.apply(fill_unique, axis=1)

    # Stufe 2 & 3: falls noch NaNs, mit mean
    for group_cols in groupings[1:]:
        fuel_map_mean = (
            df.dropna(subset=[target_col])
              .groupby(group_cols)[target_col]
              .mean()
              .round(2)
        )

        def fill_mean(row):
            if pd.isna(row[target_col]):
                key = tuple(row[col] for col in group_cols)
                return fuel_map_mean.get(key, np.nan)
            return row[target_col]

        df[target_col] = df.apply(fill_mean, axis=1)

    return df

def preprocess_new():

    # load the data 
    df = pd.read_csv('data.csv')

    # entferne Duplikate  
    df = df.drop_duplicates(subset= ['brand', 'model', 'color', 'registration_date', 'year',
    'price_in_euro', 'power_kw', 'power_ps', 'transmission_type',
    'fuel_type', 'fuel_consumption_l_100km', 'fuel_consumption_g_km',
    'mileage_in_km', 'offer_description']).reset_index(drop= True).copy()

    # Droppe zweite Index Spalte
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

     # Zeilen mit falschen Jahreszahlen werden herausgenommen
    yearsToFilter = list(df['year'].unique()[:29])
    filt = [val in yearsToFilter for val in df['year']]
    df = df[filt]

    df = df.loc[df['fuel_type'].isin(['Diesel', 'Petrol'])].copy()
    # Verbleibende Fahrzeuge die zwar Petrol und Diesel sind aber nicht g/km haben sondern 
    # Reichweite etc. 
    df = df.loc[df['fuel_consumption_g_km'].str.contains(r'g/km', na=False)]

    # Try to fix rows where brand = model
    df = fix_model_brand_conflicts(df)
    # convert to numeric
    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_fuel_consumption)
    df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_fuel_consumption_g) 

    # calculate fuel_consumption using formula
    df['fuel_consumption_l_100km'] = df.apply(calculate_fuel_consumption, axis=1)


    df = impute_fuel_staged(df, target_col='fuel_consumption_l_100km')
    df = impute_fuel_staged(df, target_col='fuel_consumption_g_km')

    df.drop(columns=['fuel_consumption_g_km'], axis = 1)
        
    # Spalten ins numerische umwandeln
    for col in ['power_ps', 'power_kw', 'mileage_in_km', 'price_in_euro']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        
    # Encoding vom Datum ins numerische
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month
    df['registration_year'] = df['registration_date'].dt.year

    df = df.drop(['registration_date', 'year','power_kw', 'fuel_consumption_g_km'], axis=1) # year sonst zweimal drinne


    df = df.dropna().reset_index(drop= True)

    return df