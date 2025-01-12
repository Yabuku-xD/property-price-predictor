import optuna
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def perform_bayesian_optimization(X_train, y_train, X_test, y_test):
    def objective(trial):
        # Define the hyperparameter search space
        alpha = trial.suggest_float("alpha", 0.0001, 10, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

        # Create the ElasticNet model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

        # Perform 5-fold cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        mean_cv_score = -scores.mean()  # Convert to positive MSE for optimization

        return mean_cv_score

    # Perform Bayesian Optimization
    study = optuna.create_study(direction="minimize")  # Minimize MSE
    study.optimize(objective, n_trials=50)  # Run for 50 trials

    # Best hyperparameters and model evaluation
    best_params = study.best_params
    best_model = ElasticNet(
        alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"], max_iter=10000
    )
    best_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Bayesian Optimization Best Hyperparameters: {best_params}")
    print(f"Bayesian Optimization Optimized Model Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    return best_model, best_params
