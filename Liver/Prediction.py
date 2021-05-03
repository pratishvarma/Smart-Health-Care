# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:27 2021

@author: ladduu
"""

import numpy as np
import joblib


# opening a pickle file for the classifier
filename = 'model_liver.sav'
classifier = joblib.load(filename)


# for preddiction
def prediction(x):
    x=np.asarray(x).reshape(1,-1)
    result = classifier.predict_proba(x)
    if result[0][0] > 0.5:
        return "congrats our model say he is {}% sure you dont have liver disease".format(int(result[0][0]*100))
    elif result[0][1]>0.5:
        return "sorry to say but our model say there is {} % chance you have a liver disease".format(int(result[0][1]*100))
    else:
        return " sorry to say our model is not sure about you,its 50-50 condition"

value = [60,0,5.2,2.4,168,126,202,6.8,2.9,0.7]
print(prediction(value))