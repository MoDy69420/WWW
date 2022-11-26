import tensorflow as tf
import numpy as np
import os
import re

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model_pokemon.h5")

def identify(model: tf.keras.Sequential, pokemons: list, img: np.ndarray) -> str:
    pred = model.predict(img)
    idx = np.argmax(pred, axis=1).tolist()[0]

    pred_name = pokemons[idx]

      # check if the name doesnt contain english characters
    if re.search('[a-zA-Z]', pred_name) == None:
        # remove all white space

        pred_name = pred_name.replace(" ", "")

    return pred_name

@app.route('/')
def home():
    return "Hello World"

@app.route('/', methods=['POST'])
def main():
    try:
        if request.method == "POST":
            data = request.json
          
            pokemons = None
          
            if data['language'] == 'ja':
                pokemons = eval(os.environ['POKEMONS_JA'])

            elif data['language'] == 'en':
                pokemons = eval(os.environ['POKEMONS'])

            pokedata = np.array(data['array'])
          
            name = identify(model, pokemons, pokedata)
          
            return name
          
    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
