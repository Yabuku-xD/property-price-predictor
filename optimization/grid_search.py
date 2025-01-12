from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def perform_elasticnet_grid_search(X_train, y_train, X_test, y_test):
    model = ElasticNet(max_iter=10000)

    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"ElasticNet Regression Best Hyperparameters: {best_params}")
    print(f"ElasticNet Regression Optimized Model Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    return best_model, best_params