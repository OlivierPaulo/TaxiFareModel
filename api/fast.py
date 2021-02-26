# write some code for the API here
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello Batch #555 Lisbon"}


@app.get("/predict_dummy")
def predict_dummy(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    return {
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }


@app.get("/predict_fare")
def predict_fare(key :str, pickup_datetime :str, pickup_longitude :float, pickup_latitude :float, dropoff_longitude :float, dropoff_latitude :float, passenger_count :int) -> dict:
    
    ## Create DataFrame with the API inputs
    X_pred = pd.DataFrame(np.array([[key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]]) , columns=['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])
    # print(X_pred)
    # print(X_pred.shape)

    ### Loading the model with the pipeline ### 
    pipeline = joblib.load("model.joblib")

    ### Return the prediction from the pipeline ###
    return { "prediction" : float(pipeline.predict(X_pred)) }

if __name__ == "__main__":

    print(predict_fare("2012-10-06 12:10:20.0000001", "2012-10-06 12:10:20 UTC", 40.7614327, -73.9798156, 40.6513111, -73.8803331, 2))
