
from Preprocessing.preprocessing_pipeline_impute_fuel_types_no_drop import preprocessing_pipeline
from Preprocessing.preprocessing_pipeline_segment import preprocessing_pipeline_segment
import pandas as pd 



def get_full_data(path='../../../data.csv', segment=None, fuel_type=None):
    if fuel_type is None:
        fuel_type = ['Diesel', 'Petrol', 'Hybrid', 'LPG', 'Electric', 'Diesel Hybrid', 'Other', 'Unknown', 'Ethanol', 'CNG', 'Hydrogen']

    df = pd.read_csv(path)

    yearsToFilter = list(df['year'].unique()[:29])  # Scraping-Fehler
    df = df[df['year'].isin(yearsToFilter)]

    df = df[df['fuel_type'].isin(fuel_type)].reset_index(drop=True)

    df = df.drop_duplicates(subset=[
        'brand', 'model', 'color', 'registration_date', 'year',
        'price_in_euro', 'power_kw', 'power_ps', 'transmission_type',
        'fuel_type', 'fuel_consumption_l_100km', 'fuel_consumption_g_km',
        'mileage_in_km', 'offer_description'])

    if segment:
        df = preprocessing_pipeline_segment(df)

    df = preprocessing_pipeline(df)

    X = df.drop(columns='price_in_euro')
    y = df['price_in_euro']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, categorical_features, numeric_features