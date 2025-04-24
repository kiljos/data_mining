import pandas as pd
import numpy as np
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
        
    # entferne Duplikate  
    df = df.drop_duplicates(subset= ['brand', 'model', 'color', 'registration_date', 'year',
       'price_in_euro', 'power_kw', 'power_ps', 'transmission_type',
       'fuel_type', 'fuel_consumption_l_100km', 'fuel_consumption_g_km',
       'mileage_in_km', 'offer_description']) 

    # Droppe zweite Index Spalte
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # andere fuel Types als Diesel und Petrol in einen anderen Datensatz extrakhieren    
    valid_fuel_types = ['Hybrid', 'Diesel Hybrid', 'Electric', 'LPG', 'CNG', 'Ethanol', 'Hydrogen', 'Other']
    df_before_filter = df.copy()    
    df_other_fuel_types = df_before_filter[(~df_before_filter['fuel_type'].isin(['Diesel', 'Petrol'])) & (df_before_filter['fuel_type'].isin(valid_fuel_types))].reset_index(drop=True)
       # evtl. den ersten Teil  ~df_before_filter['fuel_type'].isin(['Diesel', 'Petrol'])) & rausnehmen? Denn ist ja doppelt, denn Bedingung wird ja in dem 2. Teil ja schon überpüft wird
        
    df = df.loc[df['fuel_type'].isin(['Diesel', 'Petrol'])]
    df = df.loc[df['fuel_consumption_g_km'].str.contains(r'g/km', na=False)] # hiermit werden hybride Fahrzeuge rausgefiltert (haben Reichweite in g/km drin, aber trotzdem fuel Type Petrol/ Diesel
    df = df.reset_index(drop=True)
    
            
    # Zeilen mit falschen Jahreszahlen werden herausgenommen
    yearsToFilter = list(df['year'].unique()[:29])
    filt = [val in yearsToFilter for val in df['year']]
    df = df[filt]

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
    df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_fuel_consumption_g)      


    # Funktion zur Berechnung fehlender l/100km Werte, wenn g/km gegeben ist
    def calculate_fuel_consumption(row):

        conversion_factor = 0.043103448275862

        if pd.isna(row['fuel_consumption_l_100km']) or row['fuel_consumption_l_100km'] == 0:
            if pd.notna(row['fuel_consumption_g_km']) and row['fuel_consumption_g_km'] != 0:
                return row['fuel_consumption_g_km'] * conversion_factor
            else:
                return np.nan
        else:
            return row['fuel_consumption_l_100km']

    df['fuel_consumption_l_100km'] = df.apply(calculate_fuel_consumption, axis=1)
    
    # Fixe wo model = brand, versuche eindeutig Model zuzuweisen sonst droppen 
    df = fix_model_brand_conflicts(df) 

    df.drop(columns=['fuel_consumption_g_km'])
        
    # Spalten ins numerische umwandeln
    for col in ['power_ps', 'power_kw']:
            df[col] = df[col].astype(float)
    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')
        
    # Encoding vom Datum ins numerische
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month
    df['registration_year'] = df['registration_date'].dt.year

    df = df.drop(['registration_date', 'year'], axis=1) # year sonst zweimal drinne
    
    # Droppe alle Zeilen, in denen null values vorkommen
    df = df.dropna()

    # Outlier müssen wir nochmal nachfragen in der Coaching Session
    ''''
    # Outlier Detection für fuel
    q1_fuel= df['fuel_consumption_l_100km'].quantile(0.25)
    q3_fuel=df['fuel_consumption_l_100km'].quantile(0.75)
    iqr_fuel=q3_fuel-q1_fuel    
    lower_bound_fuel= q1_fuel - 1.5 * iqr_fuel
    upper_bound_fuel=22.0 # hier bisher Fixwert, da viele der Luxusautos wie Bentleys/ Lamborghini Aventadors etc. weit über q3_fuel + 1.5*iqr (der wäre 9.5)
    df = df[(df['fuel_consumption_l_100km'] >= lower_bound_fuel) & (df['fuel_consumption_l_100km'] <= upper_bound_fuel)] # damit werden ca. 240 Zeilen gelöscht
    
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
    

    # habe nochmal überlegt ist das net data leakage? Trainingsdatensatz ehält info wie der Testdatensatz verteilt ist
    
    df['outlier_model_price'] = detect_outliers_iqr(df, ['brand', 'model'], 'price_in_euro')
    df['outlier_model_mileage'] = detect_outliers_iqr(df, ['brand', 'model'], 'mileage_in_km')

    # Entferne alle Rows, die bei Preis & Mileage ein Outlier sind
    df = df[
        (~df['outlier_model_price']) &
        (~df['outlier_model_mileage'])
    ].copy()
    '''
    

    def filter_models_with_min_count(df, model_col='model', min_count=100):
        '''Diese Funktion filtert den DF auf die Modelle ein die in einer bestimmten
        Frequenz vorkommen'''
        model_counts = df[model_col].value_counts()
        models_over_threshold = model_counts[model_counts >= min_count].index
        df_filtered = df[df[model_col].isin(models_over_threshold)]
        return df_filtered
    
    # Nur die Modelle, die mind. 100 mal vorkommen
    df = filter_models_with_min_count(df, 'model', 100)
   
    return df 

