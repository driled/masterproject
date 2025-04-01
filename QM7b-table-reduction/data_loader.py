import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """
    Load QM7 dataset and extract features and target variables

    Parameters:
    path: Directory path containing the CSV file

    Returns:
    features: Molecular feature data
    energy_values: Continuous energy values (for visualization)
    energy_bins: Discretized energy categories (for evaluating clustering quality)
    """
    # Find CSV files in the directory
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory {path}")

    # Load the first CSV file found
    file_path = os.path.join(path, files[0])
    print(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    # Identify target variable - likely to be related to energy
    target_col = None
    potential_targets = ['energy', 'Energy', 'atomization_energy', 'atomization_E', 'E']
    for col in potential_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        # Save continuous energy values (for visualization)
        energy_values = df[target_col]
        print(f"Target variable: {target_col}")

        # Create discretized energy categories (for evaluating clustering quality)
        # Use quantiles to divide continuous values into categories
        n_bins = 4  # Divide into 4 categories: low, medium-low, medium-high, high energy
        energy_bins = pd.qcut(energy_values, n_bins, labels=False)
        print(f"Divided energy into {n_bins} categories for evaluation")
    else:
        energy_values = None
        energy_bins = None
        print("Warning: No energy-related column found")

    # Remove non-feature columns
    non_feature_cols = []
    for col in df.columns:
        if ('name' in col.lower() or 'id' in col.lower() or 'smiles' in col.lower() or
                col == target_col or 'compound' in col.lower() or 'molecule' in col.lower()):
            non_feature_cols.append(col)

    features = df.drop(columns=non_feature_cols, errors='ignore')
    print(f"Feature data shape: {features.shape}")

    return features, energy_values, energy_bins


def preprocess_data(X):
    """
    Preprocess feature data (standardization)

    Parameters:
    X: Feature data matrix

    Returns:
    X_scaled: Standardized feature data
    scaler: Standardizer object for subsequent transformations
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data standardization completed")
    return X_scaled, scaler