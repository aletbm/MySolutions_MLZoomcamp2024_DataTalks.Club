FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv flask

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

RUN flask install --deploy --system

COPY ["*.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "subscription_serving.py:app"]