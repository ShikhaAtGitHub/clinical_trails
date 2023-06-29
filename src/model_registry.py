import mlflow

run_name = 'bald-shark-910'

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name=run_name) as run:
     
    result = mlflow.register_model(
        "runs:/d134c724799141949b50e66d4503e326/model",
        "bald-shark-910"
    )

# mlflow.set_tracking_uri("postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres")

# client = mlflow.tracking.MlflowClient()
# client.create_registered_model("resilient-sow-323")
# result = client.create_model_version(
#     name="resilient-sow-323",
#     source="6c24cf7667cb41c893865d33d96b1ece/artifacts/model",
#     run_id="6c24cf7667cb41c893865d33d96b1ece"
# )