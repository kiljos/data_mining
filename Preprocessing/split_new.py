
from sklearn.model_selection import train_test_split
from Preprocessing.preprocessing_pipeline_impute_fuel_types_no_drop import preprocessing_pipeline
from Preprocessing.preprocessing_pipeline_segment import preprocessing_pipeline_segment
import pandas as pd 


def split_data(path = '../../data.csv', segment = None):

    df = pd.read_csv(path)

    yearsToFilter = list(df['year'].unique()[:29])  # wegen Scraping Fehler
    filt = [val in yearsToFilter for val in df['year']]
    df = df[filt]

    if segment: 
        df = preprocessing_pipeline_segment(df)


    counts       = df['model'].value_counts()
    rare_models  = counts[counts < 2].index           # nur 1 Zeile
    df_rare      = df[df['model'].isin(rare_models)]  # bleibt im Train
    df_common    = df[~df['model'].isin(rare_models)] # ≥2 Zeilen

    train_c, test_c = train_test_split(
        df_common,
        test_size   = 0.2,
        random_state= 42,
        stratify    = df_common['model']              
    )

    df_train = pd.concat([train_c, df_rare], ignore_index=True)
    df_test  = test_c.copy()
    df_train = preprocessing_pipeline(df_train)
    df_test = preprocessing_pipeline(df_test)

    X_train = df_train.drop(columns='price_in_euro')
    y_train = df_train['price_in_euro']
    X_test = df_test.drop(columns='price_in_euro')
    y_test = df_test['price_in_euro']

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    return X_train, X_test, y_train, y_test, categorical_features , numeric_features

