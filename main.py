from data.load_data import load_data
from data.split_data import split_data
from data.feature_engineering import feature_selection
from models.random_forest_model import evaluate_random_forest_model
from sklearn.ensemble import RandomForestRegressor

# Load data
X, y = load_data("data/housing.csv")
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Perform feature selection
selected_features = feature_selection(X, y, feature_names=feature_names, n_features_to_select=10)
print("Selected Features:", selected_features)

# Use only selected features
X_selected = X[selected_features]

# Split data
X_train, X_test, y_train, y_test = split_data(X_selected, y)

# Define optimized Random Forest model
optimized_rf_model = RandomForestRegressor(
    n_estimators=368,
    max_depth=29,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
)

# Evaluate the model
evaluate_random_forest_model(
    rf_model=optimized_rf_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=selected_features,
)

from joblib import dump

# Save the optimized model
dump(optimized_rf_model, "optimized_rf_model.joblib")
