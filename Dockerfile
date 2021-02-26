# write some code to build your image

### FROM the base image we want to build
FROM python:3.8.6-buster

### COPY file needed for the Docker images
COPY api /api
COPY TaxiFareModel /TaxiFareModel
COPY requirements.txt /requirements.txt
COPY model.joblib /model.joblib


RUN echo $GOOGLE_APPLICATION_CREDENTIALS > /.my_credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/.my_credentials
RUN touch /env.txt                                                                                                     
RUN printenv > /env.txt     

### RUN the directives to install the dependancies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

### CMD apply the command that the container should run once it has started
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT