from flask import Flask,request,jsonify,render_template
from flask import request
import pickle 
import pandas as pd
import numpy as np
import joblib

app1 = Flask(__name__)

model=joblib.load("credit.pkl")
model = pickle.load(open("credit.pkl","rb"))

@app1.route('/')
def home():
    return render_template("index1.html")

@app1.route('/predict',methods=['POST'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    print(output)
    index_target=pd.Series(["Fraud","Normal"])
    result=index_target[output]
    #result=list(result.values)
    #result=str(result)
    return render_template('index1.html', prediction_text='Predicted Transaction Type  {}'.format(result))
    

    output = str(result)
    if output>str(0.9):
        return render_template('result1.html',pred=f'You are safe.\nProbability of fraud transaction is {output}'.format(output))
    else:
       
        return render_template('result1.html',pred=f'You are not safe.\nProbability of fraud transaction is {output}'.format(ouput))


@app1.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app1.run(debug=False)