from flask import Flask,render_template,url_for,request,jsonify
import numpy as np
import pandas as pd
import joblib,pickle
from werkzeug.utils import secure_filename
import tempfile
import sys
import os
import glob
import re

app = Flask(__name__)

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/Heart_')
def Heart_():
    return render_template('Heart.html')

@app.route('/Heart',methods=['POST'])
def Heart():
    model = 'model/model_heart.sav'
    classifier = joblib.load(model)
    if request.method == 'POST':
        parameters = []
        parameters.append(request.form['age'])
        parameters.append(request.form['sex'])
        parameters.append(request.form['cp'])
        parameters.append(request.form['trestbps'])
        parameters.append(request.form['chol'])
        parameters.append(request.form['fbs'])
        parameters.append(request.form['restecg'])
        parameters.append(request.form['thalach'])
        parameters.append(request.form['exang'])
        parameters.append(request.form['oldpeak'])
        parameters.append(request.form['slope'])
        parameters.append(request.form['ca'])
        parameters.append(request.form['thal'])

        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = classifier.predict_proba(inputFeature)
        connect=5

        if my_prediction[0][0] > 0.5:
            result="congrats our model says he is {}% sure you dont have heart disease".format(int(my_prediction[0][0]*100))
            val="green"
        elif my_prediction[0][1]>0.5:
            result="sorry to say but our model says there is {} % chance you have a heart disease".format(int(my_prediction[0][1]*100))
            val="red"
        else:
            result="sorry to say our model is not sure about you,its 50-50 condition"
            val="black"
    return render_template('Heart.html',prediction = result, color=val, connect=connect)


@app.route('/Kidney_')
def Kidney_():
    return render_template('Kidney.html')

@app.route('/Kidney',methods=['POST'])
def Kidney():
    model = 'model/model_kidney.sav'
    classifier = joblib.load(model)
    if request.method == 'POST':
        parameters = []
        parameters.append(request.form['age'])
        parameters.append(request.form['bp'])
        parameters.append(request.form['sg'])
        parameters.append(request.form['al'])
        parameters.append(request.form['su'])
        parameters.append(request.form['rbc'])
        parameters.append(request.form['pc'])
        parameters.append(request.form['pcc'])
        parameters.append(request.form['ba'])
        parameters.append(request.form['bgr'])
        parameters.append(request.form['bu'])
        parameters.append(request.form['sc'])
        parameters.append(request.form['sod'])
        parameters.append(request.form['pot'])
        parameters.append(request.form['hemo'])
        parameters.append(request.form['pcv'])
        parameters.append(request.form['wc'])
        parameters.append(request.form['rc'])
        parameters.append(request.form['htn'])
        parameters.append(request.form['dm'])
        parameters.append(request.form['cad'])
        parameters.append(request.form['appet'])
        parameters.append(request.form['pe'])
        parameters.append(request.form['ane'])
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = classifier.predict_proba(inputFeature)
        connect=5
        if my_prediction[0][0] > 0.5:
            result="congrats our model says he is {}% sure you dont have kidney disease".format(int(my_prediction[0][0]*100))
            val="green"
        elif my_prediction[0][1]>0.5:
            result="sorry to say but our model says there is {} % chance you have a kidney disease".format(int(my_prediction[0][1]*100))
            val="red"
        else:
            result="sorry to say our model is not sure about you,its 50-50 condition"
            val="black"
    return render_template('Kidney.html',prediction = result,color=val,connect=connect)

@app.route('/diabetes_')
def diabetes_():
    return render_template('diabetes.html')

@app.route('/diabetes',methods=['POST'])
def diabetes():
    model = 'model/model_diabetes.sav'
    classifier = joblib.load(model)
    if request.method == 'POST':
        parameters = []
        parameters.append(request.form['Pregnancies'])
        parameters.append(request.form['Glucose'])
        parameters.append(request.form['BloodPressure'])
        parameters.append(request.form['SkinThickness'])
        parameters.append(request.form['Insulin'])
        parameters.append(request.form['BMI'])
        parameters.append(request.form['DiabetesPedigreeFunction'])
        parameters.append(request.form['Age'])
    
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = classifier.predict_proba(inputFeature)
        connect=5
        if my_prediction[0][0] > 0.5:
            result="congrats our model says he is {}% sure you dont have diabetes disease".format(int(my_prediction[0][0]*100))
            val="green"
        elif my_prediction[0][1]>0.5:
            result="sorry to say but our model says there is {} % chance you have a diabetes disease".format(int(my_prediction[0][1]*100))
            val="red"
        else:
            result="sorry to say our model is not sure about you,its 50-50 condition"
            val="black"
    return render_template('diabetes.html',prediction = result,connect=connect,color=val)

@app.route('/liver_')
def liver_():
    return render_template('liver.html')
@app.route('/liver',methods=['POST'])
def liver():
    model = 'model/model_liver.sav'
    if request.method == 'POST':
        classifier = joblib.load(model)
        parameters = []
        parameters.append(request.form['Age'])
        parameters.append(request.form['Gender'])
        parameters.append(request.form['Total_Bilirubin'])
        parameters.append(request.form['Direct_Bilirubin'])
        parameters.append(request.form['Alkaline_Phosphotase'])
        parameters.append(request.form['Alamine_Aminotransferase'])
        parameters.append(request.form['Aspartate_Aminotransferase'])
        parameters.append(request.form['Total_Protiens'])
        parameters.append(request.form['Albumin'])
        parameters.append(request.form['Albumin_and_Globulin_Ratio'])
    
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = classifier.predict_proba(inputFeature)
        connect=5
        if my_prediction[0][0] > 0.5:
            result="congrats our model says he is {}% sure you dont have liver disease".format(int(my_prediction[0][0]*100))
            val="green"
        elif my_prediction[0][1]>0.5:
            result="sorry to say but our model says there is {} % chance you have a liver disease".format(int(my_prediction[0][1]*100))
            val="red"
        else:
            result="sorry to say our model is not sure about you,its 50-50 condition"
            val="black"
    return render_template('liver.html',prediction = result,connect=connect,color=val)

@app.route('/parkirson_')
def parkirson_():
    return render_template('parkirson.html')

@app.route('/parkirson',methods=['POST'])
def parkirson():
    model = 'model/model_parkinsons.sav'
    classifier = joblib.load(model)
    if request.method == 'POST':
        parameters = []
        parameters.append(request.form['MDVP:Fo(Hz)'])
        parameters.append(request.form['MDVP:Fhi(Hz)'])
        parameters.append(request.form['MDVP:Flo(Hz)'])
        parameters.append(request.form['MDVP:Jitter(%)'])
        parameters.append(request.form['MDVP:Jitter(Abs)'])
        parameters.append(request.form['MDVP:RAP'])
        parameters.append(request.form['MDVP:PPQ'])
        parameters.append(request.form['Jitter:DDP'])
        parameters.append(request.form['MDVP:Shimmer'])
        parameters.append(request.form['MDVP:Shimmer(dB)'])
        parameters.append(request.form['Shimmer:APQ3'])
        parameters.append(request.form['Shimmer:APQ5'])
        parameters.append(request.form['MDVP:APQ'])
        parameters.append(request.form['Shimmer:DDA'])
        parameters.append(request.form['NHR'])
        parameters.append(request.form['HNR'])
        parameters.append(request.form['RPDE'])
        parameters.append(request.form['DFA'])
        parameters.append(request.form['spread1'])
        parameters.append(request.form['spread2'])
        parameters.append(request.form['D2'])
        parameters.append(request.form['PPE'])
    
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = classifier.predict_proba(inputFeature)
        connect=5
        if my_prediction[0][0] > 0.5:
            result="congrats our model says he is {}% sure you dont have parkirson disease".format(int(my_prediction[0][0]*100))
            val="green"
        elif my_prediction[0][1]>0.5:
            result="sorry to say but our model says there is {} % chance you have a parkirson disease".format(int(my_prediction[0][1]*100))
            val="red"
        else:
            result="sorry to say our model is not sure about you,its 50-50 condition"
            val="black"
    return render_template('parkirson.html',prediction = result,connect=connect, color=val)

@app.route('/COVID_')
def COVID_():
    return render_template('COVID.html')

@app.route('/COVID',methods=['POST'])
def COVID():
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model

    def allowed_image(fname):
        if not "." in fname:
            return False
        ext = fname.rsplit(".", 1)[1]
        if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
            return True
        else:
            return False

    if request.method == 'POST':
        f = request.files['file']
        fname=secure_filename(f.filename)
        if allowed_image(fname):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', fname)
            f.save(file_path)
            ##loading model
            classifier = load_model('model/keras_model.h5')
            test_image = image.load_img(file_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = (test_image.astype(np.float32) / 127.0) - 1
            result = classifier.predict(test_image)
            if result[0][0] > 0.7:
                classifier = load_model('model/model_covid19.h5')
                test_image2 = image.load_img(file_path, target_size=(224, 224))
                test_image2 = image.img_to_array(test_image2)
                test_image2 = np.expand_dims(test_image2, axis=0)
                result2 = classifier.predict(test_image2)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if result2 == 0:
                    return "Sorry to say you have tested COVID-19 Positive"
                else:
                    return "Hurray you have tested COVID-19 Negative"
            else:
                return "Sorry you uploaded wrong image!!"
    else:
        return "Something went wrong please refresh the page"

@app.route('/pneumonia_')
def pneumonia_():
    return render_template('pneumonia.html')

@app.route('/pneumonia',methods=['POST'])
def pneumonia():
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model

    def allowed_image(fname):
        if not "." in fname:
            return False
        ext = fname.rsplit(".", 1)[1]
        if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
            return True
        else:
            return False

    if request.method == 'POST':
        f = request.files['file']
        fname=secure_filename(f.filename)
        if allowed_image(fname):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', fname)
            f.save(file_path)
            ##loading model
            classifier = load_model('model/keras_model.h5')
            test_image = image.load_img(file_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = (test_image.astype(np.float32) / 127.0) - 1
            result = classifier.predict(test_image)
            if result[0][0] > 0.7:
                classifier = load_model('model/model_pneumonia.h5')
                test_image2 = image.load_img(file_path, target_size=(224, 224))
                test_image2 = image.img_to_array(test_image2)
                test_image2 = np.expand_dims(test_image2, axis=0)
                result2 = classifier.predict(test_image2)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if result2 == 1:
                    return "Sorry to say you have Pneumonia"
                else:
                    return "Congratulations you dont have pneumonia"
            else:
                return "Sorry you uploaded wrong image!!"
    else:
        return "Something went wrong please refresh the page"
@app.route('/lung_')
def lung_():
    return render_template('lung.html')

@app.route('/lung',methods=['POST'])
def lung():
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model

    def allowed_image(fname):
        if not "." in fname:
            return False
        ext = fname.rsplit(".", 1)[1]
        if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
            return True
        else:
            return False

    if request.method == 'POST':
        f = request.files['file']
        fname=secure_filename(f.filename)
        if allowed_image(fname):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', fname)
            f.save(file_path)
            ##loading model
            classifier = load_model('model/keras_model.h5')
            test_image = image.load_img(file_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = (test_image.astype(np.float32) / 127.0) - 1
            result = classifier.predict(test_image)
            if result[0][0] > 0.7:
                classifier = load_model('model/model_lung.h5')
                test_image2 = image.load_img(file_path, target_size=(224, 224))
                test_image2 = image.img_to_array(test_image2)
                test_image2 = np.expand_dims(test_image2, axis=0)
                result2 = np.argmax(classifier.predict(test_image2),axis=1)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if result2 == 2:
                    return " Congratulations You are Normal"
                else:
                    return "sorry to say our model says you have maligant level of caner "
            else:
                return "Sorry you uploaded wrong image!!"
    else:
        return "Something went wrong please refresh the page"
    
@app.route('/maleria_')
def maleria_():
    return render_template('maleria.html')
@app.route('/maleria',methods=['POST'])
def maleria():    
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model
    
    def allowed_image(fname):
        if not "." in fname:
            return False
        ext = fname.rsplit(".", 1)[1]
        if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
            return True
        else:
            return False
    if request.method == 'POST':
        f = request.files['file']
        fname=secure_filename(f.filename)
        if allowed_image(fname):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', fname)
            f.save(file_path)
            ##loading model
            classifier = load_model('model/keras_model2.h5')
            test_image = image.load_img(file_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = (test_image.astype(np.float32) / 127.0) - 1
            result = classifier.predict(test_image)
            if result[0][0] > 0.7:
                classifier = load_model('model/model_maleria.h5')
                test_image2 = image.load_img(file_path, target_size=(224, 224))
                test_image2 = image.img_to_array(test_image2)
                test_image2 = np.expand_dims(test_image2, axis=0)
                result2 = classifier.predict(test_image2)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if result2 == 1:
                    return " Congratulations You dont have Malaria"
                else:
                    return "sorry to say our model is says you have Malaria "
            else:
                return "Sorry you uploaded wrong image!!"
    else:
        return "Something went wrong please refresh the page"


if __name__ == '__main__':
	app.run(debug=True)
