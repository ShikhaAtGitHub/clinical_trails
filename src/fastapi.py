from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO
import requests
import psycopg2
import datetime
import json
from uuid import uuid4
import sys
from typing import List, Tuple
from lime.lime_tabular import LimeTabularExplainer
import ast

print(sys.path)
sys.path.append(r"C:\Users\u1153220\OneDrive - IQVIA\Documents\POC\clinical_trails\src")
from monitoring import log_data

app = FastAPI()

class ExplanationResult(BaseModel):
    explanation: List[Tuple[str, float]]


class WineQuality(BaseModel):
    fixed_acidity:	float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: int
    total_sulfur_dioxide: int
    density:float
    pH:float
    sulphates:float
    alcohol:float

@app.get('/')
async def root():
    return  {"message":"Congratulations! You have landed on planet Earth."}

# Establish a connection with the PostgreSQL database
conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
)

# Create a cursor to execute SQL queries
cur = conn.cursor()

# @app.post("/explain")
# async def explain(wine: WineQuality):
#     print(type(wine))
#     explanation = []
#     endpoint = 'http://localhost:1234/invocations'
#     explainer = LimeTextExplainer(class_names=['low', 'medium', 'high'])
    
#     text_instance = '\t'.join([str(value) for value in wine.dict().values()])
#     def predictor(texts):
#         # Convert the text representation back to numerical values
#         numerical_instance = [float(val) for val in texts[0].split('/t')]

#         # Send a POST request to the prediction endpoint
#         response = requests.post(endpoint, json=numerical_instance)

#         # Extract the predictions from the response
#         predictions = response.json()
#         return predictions

#     # Generate the explanation using LIME
#     explanation = explainer.explain_instance(text_instance, predictor(text_instance), num_features=5)

#     # Print the explanation
#     print(explanation.as_list())

@app.post("/explain")
async def explain(wine: WineQuality):
    endpoint = 'http://localhost:1234/invocations'
    wine_df = pd.DataFrame(wine.dict(), index=[0])

    # Get the input features (X) from the dataframe
    X = wine_df

    # Get the X.values and feature_names
    X_values = X.values
    feature_names = X.columns.tolist()

    # Create the LimeTabularExplainer
    explainer = LimeTabularExplainer(X_values, feature_names=feature_names, class_names=['low', 'medium', 'high'])
    
    response_df = pd.DataFrame(wine, index=[0])

    # Define the function to predict using the model
    def predictor(instances):
        numerical_instance = pd.DataFrame(instances, columns=response_df.columns.tolist())
        
        response = requests.post(endpoint, json=numerical_instance)

        # Extract the predictions from the response
        predictions = response.json() 
        return predictions

    # Generate the explanation using LIME
    explanation = explainer.explain_instance(response_df.iloc[0].values.tolist(), predictor, num_features=5)

    # Print the explanation
    explanation_dict = dict(explanation.as_list())

    return explanation_dict
    

@app.post('/predict')
async def predict_species(wine: WineQuality):
    request_id = str(uuid4())
    data = wine.dict()
    data_in = [[data['fixed_acidity'], data['volatile_acidity'], data['citric_acid'], data['residual_sugar'], data['chlorides'],
                data['free_sulfur_dioxide'], data['total_sulfur_dioxide'], data['density'], data['pH'], data['sulphates'], data['alcohol']]]
    endpoint = 'http://localhost:1234/invocations'
    inference_request = {
        "dataframe_records": data_in
        }
    
    response =requests.post(endpoint, json= inference_request)
    # Store the response text in the PostgreSQL database
    prediction_text = response.text
    prediction_text = eval(prediction_text)
    idx = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    insert_query = "INSERT INTO predictions (predictions, id, input) VALUES (%s, %s, %s)"
    print('data_in', data_in, type(data_in))
    cur.execute(insert_query, (str(round(prediction_text['predictions'][0], 2)), idx, str(data_in)))
    conn.commit()

    # Close the cursor and the database connection
    cur.close()
    conn.close()

    log_data(
        request_id,
        str(data_in),
        prediction_text['predictions'][0],
        idx,
        "logs"
    )

    return {
        'prediction': prediction_text,
    }

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    s=str(file,'utf-8')
    data = StringIO(s)
    df=pd.read_csv(data)
    df = df.drop(["quality"], axis = 1)
    list_val = df.values.tolist()
    inference_request = {
        "dataframe_records": list_val
        }
    endpoint = 'http://localhost:1234/invocations'
    response =requests.post(endpoint, json= inference_request)
    print(response)
    #return df
    return response.text