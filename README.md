# Data analysis
- Document here the project: TaxiFareModel
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for TaxiFareModel in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/TaxiFareModel`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "TaxiFareModel"
git remote add origin git@github.com:{group}/TaxiFareModel.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```

# Install

Go to `https://github.com/{group}/TaxiFareModel` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/TaxiFareModel.git
cd TaxiFareModel
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```

Now that we have a performant model trained in the cloud, we will expose it to the world 🌍

We will create a **Prediction API** for our model, run it on our machine in order to make sure that everything works correctly. Then we will put it in the cloud so that everyone can play with our model!

In order to do so, today we will:
- Challenge 1 : create a **Prediction API** using **FastAPI**
- Challenge 2 : create a **Docker image** containing the environment required in order to run the code of our API + the code of our API
- Challenge 3 : push this image to **Google Cloud Run** so that it is instantiated as a **Docker container** that will run our code and allow developers all over the world to use it

### Recap from Train at Scale

- We trained our TaxiFare model in the cloud using the power of the **AI Platform**
- We saved one or more versions of our trained model in `model.joblib` files in **Google Cloud Storage**

We want to use our best performing model in order to make predictions 🚀

👌 If you prefer to do so, you may download the solution of the challenge `07-Data-Engineering/03-Train-at-scale/03-Train-taxiFare-on-gcp` 👌

👌 In this case, you will quickly retrain a model on your machine. The code of the solution trains on very few rows and will allow you to proceed with the exercises of today 👌

## Before we start - let's go through some important points

### About the version of your trained model + code

⚠️ Do not forget that we cannot load a `model.joblib` file without the code that was used in order to train it! After all, we need to be using the exact same pipeline ⚠️

👉 If the `model.joblib` that you want to use today for your **Prediction API** corresponds to the latest version of you code, no worries 🎉

👉 ...or if you choose to retrain your code locally using the solution from Train at Scale, no worries either 🎉

All in all: the solution is to use the version of your code/pipeline that was **used for the training** to run the prediction. If you don't know where to start - call a TA

### About the version of your trained model + packages

⚠️ Also, we need to make sure that the versions of the packages (`numpy`, `pandas`, `scikit-learn` and so on) used in order to train the model are going to be the same as the ones used in order to load the `model.joblib` file ⚠️

👉 This is probably not going to be a concern if you trained your model recently since the versions probably did not evolve that fast

👉 You may encounter this issue in the future if you try to load a `model.joblib` file for your **Prediction API** a few months from now. The solution is to pin the versions of the packages in your `requirements.txt`. Remember the **AI Platform RUNTIME** ? The [version of the runtime](https://cloud.google.com/ai-platform/training/docs/runtime-version-list) that you used for the training allows you to know which versions of the packages to use.

## Let's create our first Prediction API exposing our model

Do you remember having consumed an API during the Python module using the `requests` package?

Today we are going to create your own API allowing other developers to ask our model for predictions.

### First step: let's create an API endpoint and test it

👉 Let's copy the provided boilerplate inside of our project (copy the `api` directory inside of your project right next to the `TaxiFareModel` directory, **not** inside of it)

👉 Let's also copy the notebooks directory at the same location, the notebook will be one way of testing your API

👉 Let's ⚠️ **append** ⚠️ the content of the `Makefile` to the Makefile of your project

Your project should look like this (use the `tree` command):

```
.
├── TaxiFareModel
│   ├── __init__.py
│   ├── data.py
│   ├── encoders.py
│   ├── gcp.py
│   ├── params.py
│   ├── trainer.py
│   └── utils.py
├── api
│   ├── __init__.py
│   └── fast.py
├── notebooks
│   └── API usage.ipynb
├── predict.py
├── Makefile
├── MANIFEST.in
├── requirements.txt
└── setup.py
```

In `api/fast.py`, let's create a root endpoint that will welcome the developers using our API.

In order to do that, we will use **FastAPI**.

⚠️ Usually our API is going to be called from the python code inside of a notebook or a package or from any language ran in the **Back-End**. That means that the code is not located inside of a web page. In this case, no issues 👌

We want our API to be quite open and to allow developers to plug it in the code that is going to run inside of a browser: the **JavaScript** code running in the browser when a web page is displayed.

In order to allow that, we need to add some specific **CORS** boilerplate to our **FastAPI** code:

``` python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
```

👉 If we do not use the `add_middleware` method, then our API will only work when called from **Back-End** code (or similar), but not when called from the **JavaScript** code of a web page. Open this link in order to learn [more about CORS](https://fastapi.tiangolo.com/tutorial/cors/)...


**Let's create our root endpoint**

Add the below code right after *middleware*:

```python
@app.get("/")
def index():
    return {"greeting": "Hello world"}
```

The endpoint will simply return the following json content when a developer hits the root of our API : http://localhost:8000/

``` json
{
  "key": "value"
}
```

*Hint*: you may create the `make run_api` **Makefile** directive in order to run the **uvicorn** web server that will serve the API:

```python
run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
```

Once the server is started, you can play with the API either directly: http://localhost:8000/

... Or through the **Swagger** documentation: http://localhost:8000/docs (click on the endpoint you wish to play with, then hit **Try it out**)

### Receive the parameters for the prediction

We have a basic endpoint for our API but that is not very useful...

Let's create a `/predict_fare` endpoint that will be used for the predictions.

We want developers to provide the following parameters to the endpoint:
- `pickup_datetime`
- `pickup_longitude`
- `pickup_latitude`
- `dropoff_longitude`
- `dropoff_latitude`
- `passenger_count`

As a response, let's first send back the provided values in order to make sure that everything is connected correctly:

For example, when called with the following parameters: `http://127.0.0.1:8000/predict_fare?pickup_datetime=2013-07-06 17:18:00 UTC&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=1`

...the API will respond:

``` json
{
  "pickup_datetime": "2013-07-06 17:18:00 UTC",
  "pickup_longitude": "-73.950655",
  "pickup_latitude": "40.783282",
  "dropoff_longitude": "-73.984365",
  "dropoff_latitude": "40.769802",
  "passenger_count": "1"
}
```

If you don't remember how to retrieve parameters from an API call go back to the lecture slides.

### Predicting the fare amount

Now that the piping is done, let's make an actual prediction.

But first, we need to store the API parameters as an observation in an `X_pred` dataframe.

The columns should match the format of the `X_train` used in order to train the pipeline of our model. Otherwise the pipeline will output a python error...

*Hint*: the code provided for the TaxiFare model uses an additional **key** column. Its value (for example `key="2013-07-06 17:18:00.000000119"`) will not affect the model however if the column is missing the pipeline will not accept the dataframe as an input...

⚠️ Also, be careful with the order of the columns when you create the dataframe! ⚠️ **Pandas** does not care about the order of the columns but **Numpy** does and you might end up surprised by the results if you build a dataframe with an incorrect order of columns.

Now that we have created a `X_pred` dataframe, let's make a prediction.

Let's load our model either from **Google Cloud Storage** or from our local hard drive.

We just need to store the resulting prediction in our **json** response:

``` json
{
  "prediction": 1.234
}
```

*Hint*: in order to play with your API, you may either fill the parameters manually in the URL, or use the notebook provided in `notebooks/API usage.ipynb`.

⚠️ The notebook is built to query an API responding to the following URL... Maybe you will want to adapt the way the notebook works if you choose a different format for your API ⚠️

`http://127.0.0.1:8000/predict_fare/?key=2012-10-06 12:10:20.0000001&pickup_datetime=2012-10-06 12:10:20 UTC&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2`

Congratulations, you just created your first API! 🎉

Let's see how we can put this API into production so that it gets exposed to the internet 🚀
