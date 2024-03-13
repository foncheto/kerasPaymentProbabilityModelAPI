from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import keras

app = Flask(__name__)

def nn_model_predict(data:pd.DataFrame,path:str):
  print('Loading model')
  # Load the model from a .keras file
  model = keras.models.load_model(path)
  print('Model loaded')
  df_data = data
  colum_dim = df_data.shape[1]
  print(model.summary())
  df_data = df_data.to_numpy()
  df_data = df_data.reshape((-1, 1, colum_dim))
  new_data_pred = model.predict(df_data)
  return np.round(new_data_pred[:]*100,2)

# Load your Keras model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the dataframe from the POST request.
        data = request.get_json(force=True)
        # go inside the data key and get the list
        data = data['data']
        # convert the list to a dataframe
        df = pd.DataFrame(data)
        
        stat = nn_model_predict(df,'models/best_keras_model.keras')
        # return the stat as a json
        return jsonify({'stat': stat.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
