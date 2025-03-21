import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
import gc  # Garbage Collector zur Speicherverwaltung
from preprocessing import preprocessing_pipeline
from eval_call import evaluate_model



def find_optimal_k(pipeline, X_train, y_train, k_range=range(1, 21)):
    
    print("Finding optimal K value...")
    
    
    param_grid = {
        'model__n_neighbors': k_range
    }
    
    # grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cv_results['mean_test_score'] * -1, marker='o')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Mean Squared Error')
    plt.title('KNN: MSE vs K Value')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig('knn_k_optimization.png')
    
    best_k = grid_search.best_params_['model__n_neighbors']
    best_score = -grid_search.best_score_
    
    print(f"Best K: {best_k}")
    print(f"Best MSE: {best_score:.2f}")
    
    return best_k


def main():
    
    X_train, X_test, y_train, y_test, X, Y, categorical_features, numeric_features = preprocessing_pipeline()
    
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Scaling is crucial for KNN
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

    # KNN pipeline with initial K=5 
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])
    
    # Optimal K 
    """best_k = find_optimal_k(knn_pipeline, X_train, y_train)
    
    # final pipeline
    final_knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(
            n_neighbors=best_k,
            weights='distance',  
            algorithm='auto',
            n_jobs=-1  
        ))
    ])"""

    final_knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',  
            algorithm='auto',
            n_jobs=-1  
        ))
    ])
    
    
    print("\nTraining KNN model with optimal K...")
    final_knn_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_knn = final_knn_pipeline.predict(X_test)
    
    # Evaluate model
    evaluate_model(y_test, y_pred_knn, "KNN Regression")
    
 
    # Clean up memory
    gc.collect()


if __name__ == "__main__":
    main()
