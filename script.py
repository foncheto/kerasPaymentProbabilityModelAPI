from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import keras
import json
import tensorflow as tf
import tensorflow_probability as tfp

app = Flask(__name__)

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def bayes_nn_structure(colum_dim):
  new_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,colum_dim)),
 
    tf.keras.layers.Dense(60,
                                    activation='relu'),
    tf.keras.layers.Dropout(0.5),
 
    tf.keras.layers.Dense(40,
                                    activation='relu'),
    tfp.layers.DenseVariational(40,
                                      make_prior_fn=prior,
                                make_posterior_fn=posterior,
                                    activation='relu'),
    tf.keras.layers.Dropout(0.5),
 
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(colum_dim, activation='softmax'),
    tf.keras.layers.Dropout(0.1),
 
    tf.keras.layers.Dense(1,
                          activation='sigmoid'),
 
    ])
  return new_model

def bayes_nn_model_predict(df_data:pd.DataFrame,path:str,amount:int):
  print('Loading Bayes model')
  colum_dim = df_data.shape[1]
  model = bayes_nn_structure(colum_dim)
  model.load_weights(path)
  print('Bayes model loaded')
  df_data = df_data.to_numpy()
  df_data = df_data.reshape((-1, 1, colum_dim))
  data_tuple = tuple([np.round(model.predict(df_data,verbose=0)[:]*100,2) for i in range(10)])
  mean_data= np.mean(np.concatenate(data_tuple,axis=1),axis=1)[:, np.newaxis]
  min_data = np.min(np.concatenate(data_tuple,axis=1),axis=1)[:, np.newaxis]
  max_data = np.max(np.concatenate(data_tuple,axis=1),axis=1)[:, np.newaxis]
 
  return np.concatenate(tuple([mean_data,min_data,max_data]),axis=1)

def nn_model_predict(data:pd.DataFrame,path:str):
  print('Loading Red model')
  # Load the model from a .keras file
  model = keras.models.load_model(path)
  print('Red model loaded')
  df_data = data
  colum_dim = df_data.shape[1]
  df_data = df_data.to_numpy()
  df_data = df_data.reshape((-1, 1, colum_dim))
  new_data_pred = model.predict(df_data)
  return np.round(new_data_pred[:]*100,2)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the dataframe from the POST request.
        data = request.get_json(force=True)
        # go inside the data key and get the list
        data = data['data']
        df = pd.DataFrame(data)
        
        red_data = nn_model_predict(df,'models/best_keras_model_14_03_2024.keras')
        bayes_data = bayes_nn_model_predict(df,'models/best_bayes_model_14_03_2024.h5',10)
        # return the stat as a json
        return jsonify({'data':{'red': red_data.tolist(), 'bayes': bayes_data.tolist()}})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)