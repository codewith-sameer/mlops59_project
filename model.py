from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=100, max_depth=3):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
