# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 21:09:01 2021

@author: ladduu
"""

from tensorflow.keras.models import load_model
##loading model
classifier = load_model('model_pneumonia.h5')


#import necessary library
import numpy as np
from tensorflow.keras.preprocessing import image

def prediction(image_path):
    test_image = image.load_img(image_path,target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = np.argmax(classifier.predict(test_image))
    print(result)
    if result == 0:
        return "congrats our model is says you dont have pneumonia"
    else:
        return "sorry to say our model is says you have pneumonia "
    
path = r"C:\Users\ladduu\Desktop\Health Care\chest_xray\chest_xray\val\person1950_bacteria_4881.jpeg"
print(prediction(path))