from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            input = request.json
            print('raw: ', [input])
            input_sc = sc.transform([input])
            print('sc: ', input_sc)
            prediction = classifier.predict(input_sc)
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Model not found')
        return('Model not found')

if __name__ == '__main__':
    classifier = joblib.load("model.pkl")
    sc = joblib.load("sc.pkl")
    print ('Model loaded')
    app.run(port=5050, debug=True)