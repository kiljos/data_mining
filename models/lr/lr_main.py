import pandas as pd
import numpy as np




from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


import sys
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline
from Preprocessing.preprocessing_pipeline_segment import preprocessing_pipeline_segment
from Preprocessing.split import split_data
from eval_call import evaluate_model



def main():
    df = preprocessing_pipeline('data.csv') 
    #df = preprocessing_pipeline_segment(df)
    X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = split_data(df)

    # Preprocessing-Pipelines erstellen
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
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