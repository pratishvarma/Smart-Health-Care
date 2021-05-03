# -*- coding: utf-8 -*-
"""
Created on Sat May  1 13:22:22 2021

@author: ladduu
"""
from tensorflow.keras.models import load_model
##loading model
classifier = load_model('model_lung.h5')


#import necessary library
import numpy as np
from tensorflow.keras.preprocessing import image

def prediction(image_path):
    test_image = image.load_img(image_path,target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = np.argmax(classifier.predict(test_image),axis=1)
    print(result)
    if result == 2:
        return " our model says you have beginning level of caner"
    elif result == 1:
        return "sorry to say our model says you have maligant level of caner "
    else:
        return "you are normal"
    
path = r"C:\Users\ladduu\Desktop\Health Care\lung cancer\Bengin case (110).jpg"
print(prediction(path))