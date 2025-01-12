import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

def preprocess_and_engineer_features(data_path):
    # Define column names
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    
    # Load the dataset
    data = pd.read_csv(data_path, header=None, names=columns, delim_whitespace=True)
    
    # Handle skewness: Apply log transformation to skewed features
    skewed_features = ["CRIM", "ZN", "DIS"]
    for feature in skewed_features:
        data[feature] = np.log1p(data[feature])
    
    # Feature engineering: Add polynomial and interaction features
    data["RM^2"] = data["RM"] ** 2
    data["LSTAT^2"] = data["LSTAT"] ** 2
    data["RM*LSTAT"] = data["RM"] * data["LSTAT"]
    
    # Handle outliers: Cap extreme values
    data["RM"] = np.clip(data["RM"], a_min=None, a_max=8.5)
    data["LSTAT"] = np.clip(data["LSTAT"], a_min=None, a_max=30)
    data["MEDV"] = np.clip(data["MEDV"], a_min=None, a_max=50)
    
    # Separate features and target
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


def feature_selection(X, y, feature_names=None, n_features_to_select=10):

    rf = RandomForestRegressor(random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    # Handle feature names
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[rfe.support_].tolist()
    else:
        if feature_names is None:
            raise ValueError("Feature names must be provided when X is a NumPy array.")
        selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]

    return selected_features
