"""
Machine Learning model creation and training functions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers
import config


def train_ml_model(model, df: pd.DataFrame, standard_scaler: bool = False):
    """
    Train a machine learning model on the given dataframe.
    
    Args:
        model: Scikit-learn compatible model
        df: DataFrame with 'Centralized_Score' as target
        standard_scaler: Whether to apply standard scaling
        
    Returns:
        Trained model
    """
    target = df['Centralized_Score']
    df_features = df.drop('Centralized_Score', axis=1)
    
    if standard_scaler:
        sc_X = StandardScaler()
        df_features = sc_X.fit_transform(df_features)
    
    model.fit(df_features, target)
    return model


def create_random_forest(n_estimators: int = None) -> RandomForestRegressor:
    """
    Create a Random Forest regressor.
    
    Args:
        n_estimators: Number of trees (default from config)
        
    Returns:
        RandomForestRegressor instance
    """
    if n_estimators is None:
        n_estimators = config.RANDOM_FOREST_N_ESTIMATORS
    return RandomForestRegressor(n_estimators=n_estimators)


def create_decision_tree() -> DecisionTreeRegressor:
    """
    Create a Decision Tree regressor.
    
    Returns:
        DecisionTreeRegressor instance
    """
    return DecisionTreeRegressor()


def create_svr() -> SVR:
    """
    Create a Support Vector Regressor.
    
    Returns:
        SVR instance
    """
    return SVR()


def create_linear_regression() -> LinearRegression:
    """
    Create a Linear Regression model.
    
    Returns:
        LinearRegression instance
    """
    return LinearRegression()


def create_polynomial_x2() -> Pipeline:
    """
    Create a polynomial regression model with x^2 features.
    
    Returns:
        Pipeline with PolynomialFeatures (degree=2) and LinearRegression
    """
    return Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])


def create_polynomial_x3() -> Pipeline:
    """
    Create a polynomial regression model with x^3 features.
    
    Returns:
        Pipeline with PolynomialFeatures (degree=3) and LinearRegression
    """
    return Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])


def create_dnn(input_shape: int, standard_scaler: bool = False) -> keras.Model:
    """
    Create a Deep Neural Network model (sequential architecture).
    
    Architecture as per paper: 632 -> 300 -> 150 -> 50 -> 1
    
    Args:
        input_shape: Number of input features
        standard_scaler: Whether to use standard scaling (not implemented in model)
        
    Returns:
        Compiled Keras model
    """
    model = keras.models.Sequential()
    
    model.add(keras.Input(shape=(input_shape,)))
    
    # Add hidden layers as per paper
    for units in config.DNN_HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation="relu"))
    
    # Output layer
    model.add(layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.DNN_LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()],
    )
    
    return model


def train_dnn(
    model: keras.Model,
    df: pd.DataFrame,
    standard_scaler: bool = False,
    batch_size: int = None,
    epochs: int = None
) -> keras.Model:
    """
    Train a DNN model.
    
    Args:
        model: Keras model
        df: DataFrame with 'Centralized_Score' as target
        standard_scaler: Whether to apply standard scaling
        batch_size: Batch size (default from config)
        epochs: Number of epochs (default from config)
        
    Returns:
        Trained model
    """
    if batch_size is None:
        batch_size = config.DNN_BATCH_SIZE
    if epochs is None:
        epochs = config.DNN_EPOCHS
    
    target = df['Centralized_Score']
    df_features = df.drop('Centralized_Score', axis=1)
    X_train = df_features.values.astype('float32')
    y_train = target.values.astype('float32')
    
    if standard_scaler:
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
    
    # Use first 10 samples for validation
    x_val = X_train[0:10]
    y_val = y_train[0:10]
    
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=0
    )
    
    return model
