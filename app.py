from pandas._config.config import reset_option
import numpy as np
from flask import Flask, request, jsonify, render_template,flash
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler

import ctypes  # An included library with Python install.   
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    surface = int(request.form["surface"])
    nbr_pieces = int(request.form["nbr_pieces"])
    langitude = float(request.form["langitude"])
    latitude = float(request.form["latitude"])
    distance_mer = min (find_nearest(latitude, langitude))
    print(distance_mer)

    if float (request.form.get('surface')) < 200 :
        x = pd.DataFrame(
                {
                    "surface_reelle_bati": [surface],
                    "nombre_pieces_principales": [nbr_pieces],
                    "longitude": [langitude],
                    "latitude": [latitude],
                    "distance_mer": [distance_mer],
                }
            )
       
        prediction = model.predict(x)  
        print(prediction)
        output = round(prediction[0],3)
        return render_template('index.html', prediction_text='Your Apartement Salary should be : {}'.format(output))
    else:
       flash('Votre Surface ne doit pas  dépasser les 200 m² !!')
       return render_template('index.html')
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

dataset2 = pd.read_csv('mer positions.csv')

def dist(lat1, long1, lat2, long2):
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def find_nearest(lat, long):
    distances = dataset2.apply(lambda row: dist(lat, long, row['latitude'], row['longitude']), axis=1)
    return distances    

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'


    app.debug = True
    app.run(debug=True)