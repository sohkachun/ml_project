from flask import Flask, request, render_template
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import  StandardScaler
from source.pipeline.predict_pipeline import PredictPipeline, CustomData


application = Flask(__name__)

app = application

## Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', "POST"])
def predict_datappoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity=  request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))

        )

        pred_df = data.get_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        resutls = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = resutls[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")