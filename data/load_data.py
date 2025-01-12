import pandas as pd

def load_data(file_path):
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    data = pd.read_csv(file_path, header=None, names=columns, sep=r"\s+")
    
    # Separate features and target
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    
    return X, y
