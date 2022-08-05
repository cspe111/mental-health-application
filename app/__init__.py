from flask import Flask, render_template, request
from app import models
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')
@app.route('/manager')
def manager():
    return render_template('manager.html')
@app.route('/check-in')
def checkin():
    return render_template('check-in.html')
@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/employee')
def index():
    return render_template('employee.html')

@app.rout('/stats')
def stats():
    return render_template('stats.html')


# Function that makes prediction
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    results = loaded_model.predict(to_predict)
    return results[0]

# Function that returns prediction based on user input
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        results = ValuePredictor(to_predict_list)

        if int(results) == 0:
            prediction = 'A conversation with your manager could benefit you!'
        else:
            prediction = 'A conversation with your manager is not necessary right now.'

        return render_template("employee.html", prediction=prediction)



if __name__ == '__main__':
    app.run(debug=False)
