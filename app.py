from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import requests
import urllib.parse
import pickle

app = Flask(__name__)
model = pickle.load(open('delhi_trained_model.pkl', 'rb'))

def want_add(add):
    if (6 > len(str(add))) or (len(str(add)) > 7):
        return None
    else:
        address = add
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
        response = requests.get(url).json()
        latitude = response[0]["lat"]
        longitude = response[0]["lon"]
        return [latitude, longitude]
    

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route('/predict', methods=["POST"])
def predict():

    try:
        features = [x for x in request.form.values()]
        address = features[1]
        int_features = features[:1] + want_add(address) + features[2:]
        print(int_features)
        final_features = [np.array(int_features)]
        my_pred = model.predict(final_features)
        
        in_lack = float(my_pred/100_000)
        output = round(in_lack, 2)
        return render_template('index.html', prediction_text='The House Price should be ₹ {} Lacks.'.format(output))
        
    except:
        return render_template('index.html', prediction_text="Invalid Input! Can not predict!")

if __name__ == "__main__":
    app.run(debug=True)