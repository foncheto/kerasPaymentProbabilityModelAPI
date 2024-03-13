from flask import Flask, request, jsonify
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

# Load your Keras model
model = load_model('path/to/your/model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data['data'])

        # Preprocess your DataFrame as needed before making predictions

        # Make predictions using your Keras model
        predictions = model.predict(df)

        # Assuming your model outputs probabilities, you can return them
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
