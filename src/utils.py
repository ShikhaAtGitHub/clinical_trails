import mlflow
from typing import Dict, Tuple
import yaml

def setup_mlflow(mlflow_config):

    # Set up the mlflow tracking server
    if mlflow_config['mlflow_artifact_uri'] is None:
        artifact_location = None
    else:
        artifact_location = mlflow_config['mlflow_artifact_uri']+"/"+mlflow_config['mlflow_experiment_name']

    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])

    try:
        experiment_id=mlflow.create_experiment(
            name=mlflow_config['mlflow_experiment_name'],
            artifact_location=artifact_location
        )
    except:
        mlflow.set_experiment(
            experiment_name=mlflow_config['mlflow_experiment_name'])
        experiment_id=mlflow.get_experiment_by_name(
            mlflow_config['mlflow_experiment_name']).experiment_id

    return experiment_id

def load_config(config_path: str) -> Tuple[int, Dict[str, str]]:
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config