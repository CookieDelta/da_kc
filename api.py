from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import uvicorn
from typing import Optional


app = FastAPI()


def load_wine_data():
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data
    y = data.target
    return X, y


class PredictionRequest(BaseModel):
    text: str
    candidate_labels: list

# Obtener detalles por experiment_id
def get_experiment_details(experiment_id: str):
    experiment = mlflow.get_experiment(experiment_id)
    if experiment:
        return {"experiment_id": experiment_id, "experiment_name": experiment.name, "artifact_location": experiment.artifact_location}
    else:
        return {"error": "Experiment not found"}

# Entrenar modelo y log MLflow
@app.post("/train_model")
def train_model(test_size: float = 0.2, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth))
    ])
 
    experiment_name = "Wine_Classifier_Experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    
    mlflow.set_experiment(experiment_id)

    mlflow.start_run()
    
    pipeline.fit(X_train, y_train)
    
    accuracy_train = accuracy_score(y_train, pipeline.predict(X_train))
    accuracy_test = accuracy_score(y_test, pipeline.predict(X_test))

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy_train', accuracy_train)
    mlflow.log_metric('accuracy_test', accuracy_test)
    mlflow.sklearn.log_model(pipeline, "wine_model")
    
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    return {
        "status": "Model trained and logged successfully",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test
    }


@app.get("/get_experiment/{experiment_id}")
def get_experiment(experiment_id: str):
    return get_experiment_details(experiment_id)

@app.get("/get_model_params")
def get_model_params(experiment_id: str):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    run_details = []
    for run in runs.itertuples():
        run_details.append({
            "run_id": run.run_id,
            "params": run.params
        })
    return {"experiment_id": experiment_id, "runs": run_details}

@app.get("/get_model_metrics")
def get_model_metrics(experiment_id: str):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    metrics_details = []
    for run in runs.itertuples():
        metrics_details.append({
            "run_id": run.run_id,
            "accuracy_train": run.metrics.get('accuracy_train', 'N/A'),
            "accuracy_test": run.metrics.get('accuracy_test', 'N/A')
        })
    return {"experiment_id": experiment_id, "metrics": metrics_details}

@app.get('/sentiment')
def sentiment_classifier(query): 
    sentiment_pipeline = Pipeline('sentiment-analysis')
    return sentiment_pipeline(query)[0]['label']

@app.get("/list_runs/{experiment_id}")
def list_runs(experiment_id: str):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    run_details = []
    for run in runs.itertuples():
        run_details.append({
            "run_id": run.run_id,
            "start_time": run.start_time,
            "status": run.status,
            "accuracy_train": run.metrics.get('accuracy_train', 'N/A'),
            "accuracy_test": run.metrics.get('accuracy_test', 'N/A'),
            "params": run.params
        })
    return {"experiment_id": experiment_id, "runs": run_details}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
