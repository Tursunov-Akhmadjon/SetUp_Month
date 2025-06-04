from flask import Flask, request, render_template
import numpy as np
import pandas as pd

application =Flask(__name__)

app =application

@app.route('/')
def index():
    return render_template('templates\index.html')

@app.route('/ predicdata', method=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        return CustomData