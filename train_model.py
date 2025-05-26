from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(df, apply_bayesian=False):
    features = ["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]
    X = df[features]
    y = df["Label"]

    # Drop rows with any missing values
    X = X.dropna()
    y = y.loc[X.index]

    if len(X) < 10:
        raise ValueError("Not enough training samples after preprocessing.")

    param_grid = {
        "n_estimators": [50],
        "max_depth": [5],
        "min_samples_split": [2],
    }

    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(X, y)
    return clf.best_estimator_
