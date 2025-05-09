from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline



#res = preprocessing_pipeline()

#print(res)



import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import gc  # Garbage Collector zur Speicherverwaltung
#from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline
from Preprocessing.preprocessing_pipeline_segment import preprocessing_pipeline_segment
from Preprocessing.split import split_data
from eval_call import evaluate_model



def main():
    df = preprocessing_pipeline() 
    #df = preprocessing_pipeline_segment(df)

    X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = split_data(df)
    print(X_train.head(5))

    # Preprocessing-Pipelines erstellen
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])


    linear_regression_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Modell trainieren
    linear_regression_pipeline.fit(X_train, y_train)


    # Vorhersagen treffen
    y_pred_lr = linear_regression_pipeline.predict(X_test)



    evaluate_model(y_test, y_pred_lr, "Linear Regression")



if __name__ == "__main__":
    main()