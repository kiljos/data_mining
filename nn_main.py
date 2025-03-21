import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor


import gc
from preprocessing import preprocessing_pipeline
from eval_call import evaluate_model





def plot_learning_curve(model, title):
    """Plot the learning curve of the neural network."""
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('nn_learning_curve.png')
    print(f"Learning curve saved as nn_learning_curve.png")


def optimize_hidden_layers(X_train, y_train, preprocessor):
    """Find optimal hidden layer configuration using cross-validation."""
    print("Finding optimal neural network architecture...")
    
    # Preprocess the data once for optimization
    X_train_processed = preprocessor.transform(X_train)
    
    # Define different hidden layer configurations to test
    hidden_layer_configs = [
        (50,),
        (100,),
        (50, 25),
        (100, 50),
        (100, 50, 25)
    ]
    
    results = []
    
    for config in hidden_layer_configs:
        model = MLPRegressor(
            hidden_layer_sizes=config,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
        
        
        model.fit(X_train_processed, y_train)
        
        
        if hasattr(model, 'best_loss_'):
            best_score = model.best_loss_
        else:
            best_score = model.loss_
            
        results.append((config, best_score))
        print(f"Hidden layers {config}: Loss = {best_score:.4f}")
    
    # Find the best configuration
    best_config = min(results, key=lambda x: x[1])[0]
    print(f"Best hidden layer configuration: {best_config}")
    
    return best_config


def main():
    
    X_train, X_test, y_train, y_test, X, Y, categorical_features, numeric_features = preprocessing_pipeline()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Scaling is crucial for neural networks
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
    
    # Fit  preprocessor
    preprocessor.fit(X_train)
    
    # Find optimal hidden layer confi
    best_hidden_layers = optimize_hidden_layers(X_train, y_train, preprocessor)
    
    # pipeline with optimized architecture
    nn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(
            hidden_layer_sizes=best_hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=200,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True,
            random_state=42
        ))
    ])
    
    
    print("\nTraining NN...")
    nn_pipeline.fit(X_train, y_train)
    
    # predictions
    y_pred_nn = nn_pipeline.predict(X_test)
    
    
    evaluate_model(y_test, y_pred_nn, "Neural Network")
    
    # Plot learning curve
    plot_learning_curve(nn_pipeline.named_steps['model'], "Neural Network Learning Curve")
    
    
    nn_model = nn_pipeline.named_steps['model']
    print("\nNeural Network Details:")
    print(f"Hidden layer sizes: {nn_model.hidden_layer_sizes}")
    print(f"Number of layers: {nn_model.n_layers_}")
    print(f"Number of outputs: {nn_model.n_outputs_}")
    print(f"Number of iterations: {nn_model.n_iter_}")
    print(f"Final loss: {nn_model.loss_:.4f}")
    
    
    gc.collect()


if __name__ == "__main__":
    main()
