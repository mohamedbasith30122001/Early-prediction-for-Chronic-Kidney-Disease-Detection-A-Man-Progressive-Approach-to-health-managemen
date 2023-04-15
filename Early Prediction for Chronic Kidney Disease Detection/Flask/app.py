from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('CKD.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/details", methods=['POST'])
def details():
    return render_template('details.html')


@app.route("/predict", methods=['POST'])
def predict():
    return render_template('predict.html')


@app.route("/result", methods=['POST'])
def result():
    if request.method == 'POST':
        input_features = [float(x) for x in request.form.values()]
        features_values = [np.array(input_features)]

        features_name = ['blood_urea', 'blood_glucose_random', 'anemia', 'coronary_artery_disease', 'pus_cell',
                         'red_blood_cells', 'diabetesmellitus', 'pedal_edema']

        df = pd.DataFrame(features_values, columns=features_name)

        output = model.predict(df)

        return render_template('result.html', prediction_text=output)


if __name__ == '__main__':
    app.run(debug=True)
