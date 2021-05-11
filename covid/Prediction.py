# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:22:38 2021

@author: ladduu
"""

from tensorflow.keras.models import load_model
##loading model
classifier = load_model('keras_model.h5')
#import necessary library
import numpy as np
from tensorflow.keras.preprocessing import image

path = r"C:\Users\ladduu\Desktop\Health Care\covid-19\covid_case.png"
test_image = image.load_img(path,target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
test_image = (test_image.astype(np.float32) / 127.0) - 1
result = classifier.predict(test_image)
if result[0][0] > 0.7:
    def prediction(image_path):
        classifier = load_model('model_covid19.h5')
        test_image = image.load_img(image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        print(result)
        if result == 1:
            return "congrats our model is says you dont have covid"
        else:
            return "sorry to say our model is says you have covid "
    print(prediction(path))
else:
    print("Sorry you uploaded wrong image")



    

