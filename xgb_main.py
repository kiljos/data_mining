
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import xgboost as xgb
import gc
from preprocessing import preprocessing_pipeline
from eval_call import evaluate_model



#Decision trees --> = XGBoost model (xgb ist ein besserer Decision Tree)

def main():
    
    X_train, X_test, y_train, y_test, X, y, categorical_features, numeric_features = preprocessing_pipeline()
    
    
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

    # Create feature names list for later use in importance analysis
    feature_names = numeric_features.copy()
    
    # Get categorical encoded feature names
    preprocessor.fit(X_train)
    try:
        categorical_encoded_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(categorical_encoded_features)
    except:
        print("Warning: Could not get encoded feature names")

    # Create XGBoost pipeline
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Train model
    print("Training XGBoost model...")
    xgb_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_xgb = xgb_pipeline.predict(X_test)
    
    
    evaluate_model(y_test, y_pred_xgb, "XGBoost")
    
   
    # Clean up memory
    gc.collect()


if __name__ == "__main__":
    main()
