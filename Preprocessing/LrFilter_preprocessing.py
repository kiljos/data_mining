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

def preprocessing_pipeline_lrfilter(X_train,y_train):
        
        
  
    # Create a DataFrame from X_train and add y_train as a column
    df = X_train.copy()
    df['price_in_euro'] = y_train

    # Fixe wo model = brand, versuche eindeutig Model zuzuweisen sonst droppen 
    df = fix_model_brand_conflicts(df)

    
    
    
    
    # ps nana drop or 0 ps drop
    df = df[~((df['power_ps'].isna()) | (df['power_ps'] == 0))].reset_index(drop=True)


    df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].fillna(0.0)


    # Outlier Detection für fuel
    q1_fuel= df['fuel_consumption_l_100km'].quantile(0.15)
    q3_fuel=df['fuel_consumption_l_100km'].quantile(0.85)

    iqr_fuel = q3_fuel - q1_fuel    
    lower_bound_fuel = q1_fuel - 1.5 * iqr_fuel

    upper_bound_fuel = 25 #fix alles löschen was über 25 l/100km ist

    df = df[(df['fuel_consumption_l_100km'] >= lower_bound_fuel) & (df['fuel_consumption_l_100km'] <= upper_bound_fuel)]
    
    # Funktion Outlier detection für Mileage 
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






    df.drop(columns=['fuel_consumption_g_km','power_kw',"registration_date"], inplace=True) 
    
    df['outlier_model_mileage'] = detect_outliers_iqr(df, ['brand', 'model'], 'mileage_in_km')

    # Entferne alle Rows, die bei Preis & Mileage ein Outlier sind
    df = df[(~df['outlier_model_mileage'])].copy()
    
    # Lösche die Outlier-Spalte, da sie nicht mehr benötigt wird
    df.drop(columns=['outlier_model_mileage'], inplace=True)

    #df.drop(columns=['offer_description'], inplace=True)

    df_valid = df.copy()
    # Fit a linear regression model
    X = df_valid[['brand', 'model', 'fuel_type', 'age', 'power_ps']]
    X = pd.get_dummies(X, columns=['brand', 'model', 'fuel_type', 'age'], drop_first=True)
    y = df_valid['fuel_consumption_l_100km'].values
    model = LinearRegression().fit(X, y)

    # Calculate predicted values and residuals
    df_valid['predicted_consumption'] = model.predict(X)
    df_valid['residual'] = df_valid['fuel_consumption_l_100km'] - df_valid['predicted_consumption']

    # Calculate standard deviation of residuals
    residual_std = df_valid['residual'].std()

    # Identify outliers (residuals more than 3 standard deviations from the mean)
    threshold = 3
    df_valid['is_outlier'] = abs(df_valid['residual']) > threshold * residual_std

    # Replace fuel consumption values with predictions where marked as outliers
    df_valid.loc[df_valid['is_outlier'], 'fuel_consumption_l_100km'] = df_valid.loc[df_valid['is_outlier'], 'predicted_consumption']

    
    # Drop intermediate columns that are no longer needed
    df_valid.drop(columns=['predicted_consumption', 'residual', 'is_outlier','registration_month',"year"], inplace=True)




    
    return df_valid
