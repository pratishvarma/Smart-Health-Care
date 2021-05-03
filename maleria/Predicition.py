# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:26:08 2021

@author: ladduu
"""


from tensorflow.keras.models import load_model
##loading model
classifier = load_model('model_maleria.h5')


#import necessary library
import numpy as np
from tensorflow.keras.preprocessing import image

def prediction(image_path):
    test_image = image.load_img(image_path,target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = classifier.predict(test_image)
    if result == 1:
        return "congrats our model is says you dont have maleriya"
    else:
        return "sorry to say our model is says you have maleriya "
    
path = r"C:\Users\ladduu\Desktop\Health Care\Malariya\infected.png"
print(prediction(path))