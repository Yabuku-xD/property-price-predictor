import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def bayesian_optimization(X, y):
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        return score

    study = optuna.create_study(
        study_name="optuna_study",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)

    print(f"Best Parameters: {study.best_params}")
    print(f"Best Score: {study.best_value:.2f}")
    return study.best_params
