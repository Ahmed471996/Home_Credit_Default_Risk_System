from flask import Flask,render_template,jsonify, request

import pandas as pd
import numpy as np
import random
#import sklearn
import pickle
from Preprocessing import *


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def upload_route_summary():
    if request.method == 'POST':
       result = request.files['file']
       result = pd.read_csv(result)
       loan_ids = result['SK_ID_CURR']
       
       
    return 

@app.errorhandler(400)
def bad_request(error=None):
    message = {
            'status': 400,
            'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
