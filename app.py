import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Pakistan Population of the year you chose should be {}'.format(output))

df = pd.read_csv('overallPak.csv')



@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = ['1951','1961','1972','1981','1998','2017','2020']
    ys = ['0.50','0.75','1.00','1.25','1.50','1.75','2.00']
    axis.plot(df.Year, df.Population)
    return fig


    









if __name__ == "__main__":
    app.run(debug=True)