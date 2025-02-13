import cloudpickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify 
    
def predict_level(patient, pipe, le, cbc):
    patient = pd.DataFrame(data=patient, index=[0])
    X_test = pipe.transform(patient)
    pred = cbc.predict(X_test)
    return le.inverse_transform(np.ravel(pred))[0]

with open('../model/obesity-levels-model.bin', 'rb') as f_in:
    pipe, le, cbc = cloudpickle.load(f_in)
    
app = Flask('estimation-obesity-levels')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    prediction = predict_level(patient, pipe, le, cbc)
    result = {'obesity_level': str(prediction)}
    return jsonify(result)

@app.route('/')
def index():
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
 
 