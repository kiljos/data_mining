
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


import shap # shap.TreeExplainer
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from Preprocessing.preprocessing_pipeline_initial import preprocessing_pipeline
from Preprocessing.split import split_data
from eval_call import evaluate_model



#Decision trees --> = XGBoost model (xgb ist ein besserer Decision Tree)

def main():
    
    df = preprocessing_pipeline()
    X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features = split_data(df)
    
    
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
        # Train model
    print("Training XGBoost model...")
    xgb_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_xgb = xgb_pipeline.predict(X_test)
    
    evaluate_model(y_test, y_pred_xgb, "XGBoost")
    
    # Clean up memory
    gc.collect()
    
    # Get the preprocessed data for SHAP analysis
    X_processed = xgb_pipeline.named_steps['preprocessor'].transform(X)
    
    # Create feature names list for SHAP plot
    all_feature_names = []
    # Add numeric feature names
    all_feature_names.extend(numeric_features)
    # Add encoded categorical feature names
    for i, cat_feature in enumerate(categorical_features):
        # Since we're using OrdinalEncoder, we'll append the category name
        all_feature_names.append(cat_feature)
    
    # Create SHAP explainer with feature names
    model = xgb_pipeline.named_steps['model']
    explainer = shap.Explainer(model, feature_names=all_feature_names)
    
    # Calculate SHAP values on processed data
    shap_values = explainer(X_processed)
    
    # Visualize the first prediction's explanation with feature names
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[0])
    plt.tight_layout()
    plt.savefig("shap_waterfall.png")
    
    # For a summary plot of feature importance with feature names
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=all_feature_names)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    
    # Adding PCA analysis to see feature importance in reduced dimensions
    from sklearn.decomposition import PCA
    
    # Apply PCA on the processed data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    # Create a DataFrame to show feature importance in PCA
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=all_feature_names
    )
    
    # Sort by absolute contribution to first principal component
    pca_components['PC1_abs'] = abs(pca_components['PC1'])
    pca_sorted = pca_components.sort_values('PC1_abs', ascending=False)
    
    print("Feature importance based on PCA contribution:")
    print(pca_sorted)
    
    # Visualize PCA components
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Preprocessed Data')
    plt.savefig("pca_scatter.png")
    
    # Plot feature contributions to PCA
    plt.figure(figsize=(14, 10))
    pca_sorted.drop('PC1_abs', axis=1).plot(kind='bar')
    plt.title('Feature Contributions to Principal Components')
    plt.xlabel('Features')
    plt.ylabel('Contribution')
    plt.tight_layout()
    plt.savefig("pca_contributions.png")



if __name__ == "__main__":
    main()
