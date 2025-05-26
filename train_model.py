from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(df, apply_bayesian=False):
    features = ["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]
    X = df[features]
    y = df["Label"]

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid.fit(X, y)
    return grid.best_estimator_
