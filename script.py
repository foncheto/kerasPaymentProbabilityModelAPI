from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your Keras model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
