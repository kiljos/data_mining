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
    df = df.drop('registration_date', axis=1)

    # Droppe alle Zeilen, in denen null values vorkommen
    df = df.dropna()

    # Outlier Detection für fuel
    q1_fuel= df['fuel_consumption_l_100km'].quantile(0.25)
    q3_fuel=df['fuel_consumption_l_100km'].quantile(0.75)
    iqr_fuel=q3_fuel-q1_fuel    
    lower_bound_fuel= q1_fuel - 1.5 * iqr_fuel
    upper_bound_fuel=22.0 # hier bisher Fixwert, da viele der Luxusautos wie Bentleys/ Lamborghini Aventadors etc. weit über q3_fuel + 1.5*iqr (der wäre 9.5)
    df = df[(df['fuel_consumption_l_100km'] >= lower_bound_fuel) & (df['fuel_consumption_l_100km'] <= upper_bound_fuel)] # damit werden ca. 240 Zeilen gelöscht
    # !!! TO-DO: Noch anwenden auf Preis, power_ps  und mileage
            # !!! TO-DO: bei power_ps nur eigener unterer Threshold 
            # !!!! TO-DO: bei Kilometer eigener oberer Threshold

    def filter_models_with_min_count(df, model_col='model', min_count=100):
        '''Diese Funktion filtert den DF auf die Modelle ein die in einer bestimmten
        Frequenz vorkommen'''
        model_counts = df[model_col].value_counts()
        models_over_threshold = model_counts[model_counts >= min_count].index
        df_filtered = df[df[model_col].isin(models_over_threshold)]
        return df_filtered
    
    # Nur die Modelle, die mind. 100 mal vorkommen
    df = filter_models_with_min_count(df, 'model', 100)
    # Auswerten der Offer Description Spalte

    """ 
    Kurz zusammengefasst: Mit dem SkLearn Count Vectorizer werden die 50 häufigsten Wörter in der Spalte Offer Description
    gezählt. Weil viele davon inhaltlich irgendwie zusammengehören, z.B. Quattro, xDrive, all-wheel-drive und einige auch 
    nicht sinnvoll miteingebunden werden können, z.B. das Wort "Hand" wegen 2., 3. 4. Hand, oder "auto", wurden diese Wörter
    extrahiert und in einer .csv datei kategorisiert. 
    Diese .csv Datei namens "offer_description_utf8.csv" wird dann wieder importiert und daraus ein dictionary erstellt. 
    Mit Hilfe dieses Dictionaries wird dann für jede Kategorie eine neue Spalte erstellt und mit Yes / No ausgefüllt.
    z.B. Neue Spalte "has_navi": Yes, wenn in der Spalte offer description "navi" gefunden wurde, sonst No
    (insgesamt 17 Kategorien, also 17 neue Spalten)
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Liste von deutschen Stopwörter, die nicht mitgezählt werden sollen (generiert von ChatGPT)
    german_stopwords = [
    "aber", "abgesehen", "alle", "allem", "allen", "alles", "als", "also", "am", "an", "andere", "anderen", "anderm", 
    "anderer", "anderes", "auch", "auf", "aus", "bei", "beide", "beiden", "beim", "beiner", "beim", "bis", "bisschen", 
    "bleiben", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er", "es", "essen", "etwas", "für", "gegen", 
    "geht", "habe", "haben", "hat", "hatte", "hätten", "hattest", "hattete", "hier", "hin", "hinter", "ich", "ihn", "ihm", 
    "ihr", "ihre", "immer", "in", "indem", "ins", "irgendwas", "ist", "ja", "jeder", "jeden", "jedes", "jedoch", "kann", 
    "kein", "keine", "konnte", "konnte", "machen", "mehr", "mit", "muss", "muss", "nach", "nicht", "nichts", "nur", "ob", 
    "oder", "ohne", "sehr", "sich", "sie", "sind", "so", "sollen", "sollte", "sondern", "um", "und", "uns", "unsere", 
    "vom", "von", "vor", "während", "weil", "wer", "werden", "wie", "wieder", "wieso", "wir", "wird", "wirst", "wo", 
    "woher", "wohin", "zu", "zum", "zur", "zwar", "zwischen"
    ]

    # Entferne Sonderzeichen und Zahlen (weil zu unaussagekräftig) und wandle alles in Kleinbuchstaben um
    # Wird in die neue Spalte _cleaned geschrieben
    df["offer_description_cleaned"] = (
    df["offer_description"]
    .str.replace(r'\d+', '', regex=True)             # Entfernt alle Zahlen
    .str.replace(r'[^\w\s]', '', regex=True)         # Entfernt Sonderzeichen
    .str.lower()                                     # Alles klein
    )

    #Sollten eigentlich schon draußen sein, aber entfernt zur sicherheit nochmal alle leeren werte 
    df = df.dropna(subset=["offer_description_cleaned"]) 

    # Erstellen eines CountVectorizer, der die deutschen Stopwörter verwendet -> nimmt sich hier jetzt die 50 häufigsten wörter
    vectorizer = CountVectorizer(analyzer='word', stop_words=german_stopwords, max_features=50)

    # Anwendung des Vectorizers
    X = vectorizer.fit_transform(df["offer_description_cleaned"])

    # Erstellen eines DataFrame, um die Häufigkeit der Wörter zu sehen
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Berechnung der Häufigkeit jedes Schlagworts
    word_count = word_freq.sum().sort_values(ascending=False)



    # Die oben ausgegebene Liste an häufigsten Wörtern wurde dann per Hand aussortiert, e.g. Wörter wie "auto, matic" usw. wurden entfernt
    # Und ähnliche Wörter in Kategorien zusammengefasst z.B. "quattro, xDrive und awd = all wheel drive"
    # Einlesen der Kategorien aus der csv-datei "offer_description_utf8.csv"
    kat = pd.read_csv("offer_description_utf8.csv", sep=";")
    # Werte ohne zugeordnete Kategorien werden gelöscht
    kat = kat.dropna(subset=["Kategorie"])

    # Kategorien + zugehörige Abkürzungen in Dictionary gruppieren
    abkuerzung_dict = kat.groupby("Kategorie")["Abkuerzung"].apply(list).to_dict()

    # Für jede Kategorie eine neue Spalte erzeugen, die boolesche Werte enthält
    # Also z.B. has_navigation? yes / no
    for kategorie, abkuerzungen in abkuerzung_dict.items():
        df[kategorie] = df["offer_description"].apply(
        lambda text: True if any(abk.lower() in text.lower() for abk in abkuerzungen) else False
        )
            
     # Zielvariable festlegen       
    y = df['price_in_euro']
    X = df.drop(['price_in_euro'], axis=1)

    # Identifiziere numerische und kategoriale Spalten
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # (Hinweis: Der Limiting-Block für kategoriale Features wurde entfernt.)

    # Speicher freigeben
    gc.collect()

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['model'])
    return X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features
