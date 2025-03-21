import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import gc  # Garbage Collector zur Speicherverwaltung
from preprocessing import preprocessing_pipeline
from eval_call import evaluate_model



def main():
    X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = preprocessing_pipeline()

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