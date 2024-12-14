from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np
import os

import tflite_runtime.interpreter as tflite

MODEL_NAME = os.getenv('MODEL_NAME', 'model_2024_hairstyle_v2.tflite')

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()
		
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
		
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img
    
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
    
def prepare_input(x):
    return x / 255.0    
    
def predict(X):
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return float(preds[0,0])
    
def decode_predictions(pred):
	return ["curly", "straight"][pred > 0.5]
		
def lambda_handler(event, context):
    url = event['url']
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)
    preds = predict(X)
    results = decode_predictions(preds)
    return {"prediction" : results}