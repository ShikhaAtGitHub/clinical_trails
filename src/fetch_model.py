import pandas as pd
import mlflow.pyfunc

def fetch():
    mlflow.set_tracking_uri("http://localhost:5000")
    model_name = "bald-shark-910"
    stage = 'Production'

    X_test = pd.read_csv(r'C:\Users\u1153220\OneDrive - IQVIA\Documents\POC\clinical_trails\data\winequality-red_test.csv')
    X_test = X_test.drop(["quality"], axis=1)
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="bald-shark-910",
        version=1,
        stage="Production"
    )

    # model_uri = "runs:/d134c724799141949b50e66d4503e326/artifacts/model"
    # model = mlflow.sklearn.load_model(model_uri=model_uri)

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    y_pred = model.predict(X_test)
    print(y_pred)

fetch()