import pandas as pd
import numpy as np
import re
import gc  # Garbage Collector zur Speicherverwaltung


def preprocessing_pipeline(path = '../data.csv'):
        
    # Daten laden
    df = pd.read_csv(path)
        
    # entferne Duplikate  
  

    # Droppe zweite Index Spalte
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
   
     
    df = df.loc[df['fuel_type'].isin(['Diesel', 'Petrol'])]
    #df = df.loc[df['fuel_consumption_g_km'].str.contains(r'g/km', na=False)] # hiermit werden hybride Fahrzeuge rausgefiltert (haben Reichweite in g/km drin, aber trotzdem fuel Type Petrol/ Diesel
    df = df.reset_index(drop=True)
    
            
    # Zeilen mit falschen Jahreszahlen werden herausgenommen
    yearsToFilter = list(df['year'].unique()[:29])
    filt = [val in yearsToFilter for val in df['year']]
    df = df[filt]


    def Electrics_Reichweite(df):
    # Filtere die Zeilen, bei denen "fuel_type" = "Electric" und "fuel_consumption_g_km" das Wort "Reichweite" enthält
        e_mit_reichweite = (df["fuel_type"] == "Electric") & (df["fuel_consumption_g_km"].astype(str).str.contains("Reichweite", na=False))
        # Cutte in der Spalte "fuel_consumption_g_km" beim ersten Leerzeichen
        df.loc[e_mit_reichweite, "fuel_consumption_g_km"] = df.loc[e_mit_reichweite, "fuel_consumption_g_km"].astype(str).str.split().str[0]
        # Kopiere Werte von "fuel_consumption_g_km" in die Spalte "fuel_consumption_l_100km"
        df.loc[e_mit_reichweite, "fuel_consumption_l_100km"] = df.loc[e_mit_reichweite, "fuel_consumption_g_km"]
        
        return df
    
    df = Electrics_Reichweite(df)
                
    def clean_fuel_consumption(value): 
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

    # setze die Ausreißer bei Petrol, Diesel in fuel_consumption_100km auf nan

   

    # Spalten ins numerische umwandeln
    for col in ['power_ps', 'power_kw']:
            df[col] = df[col].astype(float)

    df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')   
    
    # Encoding vom Datum ins numerische
    df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
    df['registration_month'] = df['registration_date'].dt.month
   # df['registration_year'] = df['registration_date'].dt.year

    #df = df.drop(['registration_date','power_kw', 'offer_description', 'fuel_consumption_g_km'], axis=1)
   
    return df 
