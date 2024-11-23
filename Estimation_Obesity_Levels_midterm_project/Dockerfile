FROM python:3.11.10-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["./scripts/train.py", "./scripts/predict.py", "./scripts/"]
COPY ["./model/obesity-levels-model.bin", "./model/"]

WORKDIR /app/scripts

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]