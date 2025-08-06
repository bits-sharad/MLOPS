from flask import Flask, request, jsonify
import joblib
import mlflow
import numpy as np

app = Flask(__name__)
model = joblib.load('random_forest.joblib')

@app.route('/predict', methods=['POST'])    
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1, -1))

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
    # app.run(debug=True, host=0.0.0.0, port=5000)
    