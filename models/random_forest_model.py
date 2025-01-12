from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.visualization import plot_feature_importance

def evaluate_random_forest_model(rf_model, X_train, y_train, X_test, y_test, feature_names):
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Model Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    print("\nFeature Importance Analysis:")
    feature_importance_df = plot_feature_importance(rf_model, feature_names)
    return feature_importance_df
