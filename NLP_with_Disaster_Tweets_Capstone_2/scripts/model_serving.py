import numpy as np
import os
import sys
sys.path.append(os.path.dirname("./utils.py"))
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from utils import init_configure, preprocessing_text, preprocessing_keyword, get_tokens
import cloudpickle
import spacy

def transform_text(tweet):
    clean_text, n_words, n_characters, n_hashtags, n_mentions, n_urls, n_punctuations = preprocessing_text(tweet["text"], nlp, matcher, dict_words)
    clean_keyword = preprocessing_keyword(tweet["keyword"], nlp)

    tokenized_text = get_tokens([clean_text], tokenizer=tokenizer, max_length=max_length_tweet, fit=False, padding=True)
    tokenized_keyword = get_tokens([clean_keyword], tokenizer=tokenizer, max_length=max_length_keyword, fit=False, padding=True)
    context = np.array([n_words, n_characters, n_hashtags, n_mentions, n_urls, n_punctuations])

    tokenized_text = tokenized_text.astype(np.float32)
    tokenized_keyword = tokenized_keyword.astype(np.float32)
    context = np.expand_dims(context, axis=0).astype(np.float32)
    return tokenized_text, tokenized_keyword, context

def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

def make_request(tokenized_text, tokenized_keyword, context):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'disaster_tweets_model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['text'].CopyFrom(np_to_protobuf(tokenized_text))
    pb_request.inputs['keyword'].CopyFrom(np_to_protobuf(tokenized_keyword))
    pb_request.inputs['context'].CopyFrom(np_to_protobuf(context))
    return pb_request
    
def process_response(pb_result):
    pred = pb_result.outputs['dense_5'].float_val
    return {"classification": ["Not Disaster Tweet", "Disaster Tweet"][round(pred[0])]}

def apply_model(tweet):
    tokenized_text, tokenized_keyword, context = transform_text(tweet)
    pb_request = make_request(tokenized_text, tokenized_keyword, context)
    pb_result = stub.Predict(pb_request, timeout=40.0)
    return process_response(pb_result)

nlp = spacy.load("en_core_web_lg")
matcher, _, max_length_tweet, max_length_keyword = init_configure(nlp)
    
with open('./models/tokenizer.bin', 'rb') as f_in:
    tokenizer, dict_words = cloudpickle.load(f_in)

host = os.getenv('TF_SERVING_HOST', '172.17.0.3:8500')
channel = grpc.insecure_channel(host, options=(('grpc.enable_http_proxy', 0),))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

app = Flask('disaster_tweets_model')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.get_json()
    result = apply_model(tweet)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

#To run
#docker run -it --rm -p 8500:8500 -v "$(pwd)/scripts/disaster_tweets_model:/models/disaster_tweets_model/1" -e MODEL_NAME=disaster_tweets_model tensorflow/serving:2.14.0
