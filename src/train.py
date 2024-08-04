import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Parameters
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}

# Start MLflow run
with mlflow.start_run():
    # Perform GridSearchCV
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Predict
    predictions = best_model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    
    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(best_model, "model")

    # Save the best model locally to the models directory
    joblib.dump(best_model, "models/best_model.pkl")

print(f"Best parameters: {best_params}")
print(f"Model accuracy: {accuracy}")
