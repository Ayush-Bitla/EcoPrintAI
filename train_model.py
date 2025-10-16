import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings

DATA_PATH = 'materials_data.csv'
MODEL_PATH = 'model.pkl'

def compute_sustainability_score(df: pd.DataFrame, carbon_w=0.3, recyclability_w=0.25) -> tuple:
    inv_cf = 1.0 / np.clip(df['carbon_footprint'].values, 0.1, None)
    score = (
        df['recyclability'] * recyclability_w +
        inv_cf * carbon_w +
        df['biodegradability'] * 0.30 +
        (1.0 - df['toxicity']) * 0.25 +
        df['energy_efficiency'] * 0.20
    )
    norm_factor = max(score)
    score_norm = (score / norm_factor).clip(0, 1)
    print(f"Debug: norm_factor = {norm_factor:.3f}")
    return score_norm, norm_factor

def prepare_training_dataframe(df: pd.DataFrame) -> tuple:
    df = df.copy()
    if len(df) < 10:
        warnings.warn("Dataset is small (<10 rows).")
    for col in ['tensile_strength', 'max_temp', 'density', 'cost_per_kg', 'carbon_footprint', 'volume_cm3', 'surface_area_cm2']:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values in {col}.")
    df['sustainability_score'], norm_factor = compute_sustainability_score(df)
    mechanical_component = (
        (df['tensile_strength'] / 100.0) * 0.3 +
        ((1.0 - df['flexibility']) * 0.2) +
        (df['max_temp'] / 250.0) * 0.2 +
        (1.0 - df['cost_per_kg'] / 150.0) * 0.2 +
        (1.0 - df['volume_cm3'] / 50.0) * 0.1 +
        (1.0 - df['surface_area_cm2'] / 100.0) * 0.1
    ).clip(0, 1)
    df['overall_suitability_score'] = (mechanical_component * 0.5 + df['sustainability_score'] * 0.5).clip(0, 1)
    return df, norm_factor

def train_model():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {DATA_PATH} not found.")
    df = df.dropna()
    df, norm_factor = prepare_training_dataframe(df)
    feature_cols = [
        'tensile_strength', 'flexibility', 'max_temp', 'density', 'cost_per_kg',
        'recyclability', 'carbon_footprint', 'biodegradability', 'toxicity', 'energy_efficiency',
        'volume_cm3', 'surface_area_cm2'
    ]
    X = df[feature_cols]
    y = df['overall_suitability_score'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Validation R2: {r2:.3f} | MAE: {mae:.3f} | MSE: {mse:.3f}')
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': pipeline,
            'feature_cols': feature_cols,
            'norm_factor': norm_factor
        }, f)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    train_model()