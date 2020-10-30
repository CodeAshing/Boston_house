
from flask import Flask, request, render_template,jsonify
from keras.models import load_model
import numpy as np

app=Flask(__name__)


def get_model():
    global model
    model = load_model('boston_house.h5')
    print("Model Loaded")


def text_preprocess(state):
    value = np.float32([state['length'],state['width']])    
    mean = value.mean()
    value -= mean
    std = value.std()
    value /= std
    value = np.append(value,[1,1,1,1,1,1,1,1,1,1,1] )
    value=np.expand_dims(value, axis=0)
    return value
"""
Routes
"""
@app.route('/',methods=['GET'])

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    state = request.get_json(force=True)
    value = text_preprocess(state)
    get_model()
    result = model.predict(value)
    result=result[0,0]        
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)