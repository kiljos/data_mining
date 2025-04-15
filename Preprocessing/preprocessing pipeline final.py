def preprocessing_pipeline():
        
    # Daten laden
        df = pd.read_csv('data.csv')
        
    # entferne unnötige Spalten    
        df = df.drop_duplicates() 

        if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)

    # Autos mit fuel type Elektro, Hybrid, LPG, Ethonal Autos, CNG, Other in anderen Datensatz extrahieren
        #TO-DO: von Tayyaba einfügen!!!
        
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
        conversion_factor = 0.043103448275862

        def calculate_fuel_consumption(row):
            if pd.isna(row['fuel_consumption_l_100km']) or row['fuel_consumption_l_100km'] == 0:
                if pd.notna(row['fuel_consumption_g_km']) and row['fuel_consumption_g_km'] != 0:
                    return row['fuel_consumption_g_km'] * conversion_factor
                else:
                    return np.nan
            else:
                return row['fuel_consumption_l_100km']

        df['fuel_consumption_l_100km'] = df.apply(calculate_fuel_consumption, axis=1)

        # TO-DO: fuel consumption Spalte g/km droppen!!!!
        
        # Spalten ins numerische umwandeln
        for col in ['power_ps', 'power_kw']:
               df[col] = df[col].astype(float)
        df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
        df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')
        
        # Encoding vom Datum ins numerische
        df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%Y', errors='coerce')
        df['registration_month'] = df['registration_date'].dt.month
        df['registration_year'] = df['registration_date'].dt.year
        df = df.drop('registration_date', axis=1)


        df = df.dropna()

        # Outlier Detection einfügen! Sowohl für mileage, Preis, fuel consumption etc.
        def detect_outliers_iqr(df, group_col, target_col):
            outlier_flags = pd.Series(False, index=df.index)  # alle erstmal False
            for name, group in df.groupby(group_col):
                q1 = group[target_col].quantile(0.25)
                q3 = group[target_col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                mask = (group[target_col] < lower) | (group[target_col] > upper)
                outlier_flags.loc[group[mask].index] = True  # korrekt zuordnen

            return outlier_flags
        # !!! TO-DO: Noch anwenden auf Preis und fuel consumption
        # !!! TO-DO: bei power_ps nur eigener unterer Threshold 
        # !!!! TO-DO: bei Kilometer eigener oberer Threshold
