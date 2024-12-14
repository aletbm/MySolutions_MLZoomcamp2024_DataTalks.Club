FROM agrigorev/model-2024-hairstyle:v3

RUN pip install Pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install numpy==1.23.1

COPY hw.py .
ENV MODEL_NAME=model_2024_hairstyle_v2.tflite

CMD [ "hw.lambda_handler" ]