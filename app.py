from flask import Flask, render_template,request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app=application

# Load the pre-trained model and scaler
model = pickle.load(open('models/forest_fire_model.pkl', 'rb'))
scaler = pickle.load(open('models/forest_fire_scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        classes =float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data= scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, classes, region]])
        result = model.predict(new_data)
        
        return render_template('home.html', result=result[0])
        
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)