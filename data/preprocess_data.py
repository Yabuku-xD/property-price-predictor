from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data):
    """
    Apply preprocessing and feature engineering to the dataset.
    """
    # Handle skewness
    skewed_features = ["CRIM", "ZN", "DIS"]
    for feature in skewed_features:
        data[feature] = np.log1p(data[feature])

    # Add engineered features
    data["RM^2"] = data["RM"] ** 2
    data["LSTAT^2"] = data["LSTAT"] ** 2
    data["RM*LSTAT"] = data["RM"] * data["LSTAT"]

    # Cap extreme values
    data["RM"] = np.clip(data["RM"], a_min=None, a_max=8.5)
    data["LSTAT"] = np.clip(data["LSTAT"], a_min=None, a_max=30)
    data["MEDV"] = np.clip(data["MEDV"], a_min=None, a_max=50)

    # Separate features and target
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y