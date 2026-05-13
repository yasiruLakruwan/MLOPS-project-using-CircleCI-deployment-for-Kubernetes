from flask import Flask,render_template,request
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load("artifacts/models/model.pkl")

@app.route('/', methods=['GET','POST'])

def index():
    prediction = None

    if request.method == 'POST':
        sepel_length = float(request.form['SepalLengthCm'])
        sepel_width = float(request.form['SepalWidthCm'])
        patel_length = float(request.form['PetalLengthCm'])
        patel_width = float(request.form['PetalWidthCm'])

        input_data = np.array([[sepel_length,sepel_width,patel_length,patel_width]])

        prediction = model.predict(input_data)[0]

    return render_template('index.html',prediction=prediction)


if __name__=="__main__":
    app.run(debug=True)