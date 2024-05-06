from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import json

# loading crop image source json file
with open('static/crop_image_src.json') as user_file:
    crop_img = user_file.read()

    crop_img_src = json.loads(crop_img)


with open('static/crop_summary.json') as file:
    # Load the JSON data from the file
    crop_summary = json.load(file)

with open('static/crop_name.json') as file:
    # Load the JSON data from the file
    crop_name = json.load(file)
    


# creating flask app
app = Flask(__name__)

# loading Models
rf_model = pickle.load(open(r"RandomForest.pkl",'rb'))
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

@app.route('/crop_finder.html')
def crop_rec():
    return render_template('/crop_finder.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Nitrogen = request.form['Nitrogen']
        P = request.form['P']
        K = request.form['K']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall_In_mm = request.form['rainfall_In_mm']

        feature = np.array([[Nitrogen,P,K,temperature,humidity,ph,rainfall_In_mm]])
        
        predicted_value = rf_model.predict(feature)

    predicted_crop = str(predicted_value[0])
    
    index = 0
    for item in crop_name:
        if predicted_crop == item:
            predicted_crop_index = index
            break
        else:    
            index += 1
    predicted_crop_index = index
        
    img_src = crop_img_src[predicted_crop]

    predicted_crop_info = crop_summary[predicted_crop_index]

    return render_template('/predicted.html', predicted_value=str(predicted_value[0]) , img_src = img_src, crop_info = predicted_crop_info)


@app.route('/yield.html')
def yield_page():
    return render_template('/yield.html')

@app.route('/yield',methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        country = request.form['country']
        crop  = request.form['crop']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,country,crop]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('/yield.html',prediction=prediction)


# python main
if __name__=='__main__':
    app.run(debug=True)