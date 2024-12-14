import pickle
from flask import Flask, request, jsonify
import threading

def predict_subcription_proba(customer, dv, model):
    X_customer = dv.transform(customer)
    return model.predict_proba(X_customer)[:, 1]

app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    prediction = predict_subcription_proba(customer, dv, model)

    result = {'subscription_probability': float(prediction[0])}
    return jsonify(result)

if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':9696}).start()