FROM python:3.10.11-slim-bullseye

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["../Pipfile", "../Pipfile.lock", "./"]
RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY "../models/tokenizer.bin" "./models/tokenizer.bin"
COPY "../scripts/utils.py" "./utils.py"
COPY "../scripts/model_serving.py" "./model_serving.py"

EXPOSE 9696
EXPOSE 8500

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "model_serving:app"]

#First, you need to execute the following command:
#pipenv install flask waitress cloudpickle num2words spacy tensorflow==2.14.0 grpcio==1.68.0 tensorflow-serving-api==2.14 $(spacy info en_core_web_lg --url) numpy==1.24

#Next, we build the docker image
#docker build -t serving-gateway-disaster-tweets-model -f ./etc/gateway.dockerfile .

#To tag it
#docker tag serving-gateway-disaster-tweets-model aletbm/serving-gateway-disaster-tweets-model

#To do push it
#docker push aletbm/serving-gateway-disaster-tweets-model:latest