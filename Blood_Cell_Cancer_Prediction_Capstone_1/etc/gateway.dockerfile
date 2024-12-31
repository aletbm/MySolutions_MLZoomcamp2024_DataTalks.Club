FROM python:3.11.7-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["../Pipfile", "../Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY "../scripts/model_serving.py" "model_serving.py"

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "model_serving:app"]

#First, you need to execute the following command:
#pipenv install flask waitress pillow tensorflow==2.17.0 grpcio tensorflow-serving-api==2.17.0

#Next, we build the docker image
#docker build -t serving-gateway-blood-cell-model -f ./etc/gateway.dockerfile .

#To tag it
#docker tag serving-gateway-blood-cell-model aletbm/serving-gateway-blood-cell-model

#To do push it
#docker push aletbm/serving-gateway-blood-cell-model:latest