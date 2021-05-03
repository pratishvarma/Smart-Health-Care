# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:59:24 2021

@author: ladduu
"""

import numpy as np
import joblib


# opening a sav file for the classifier
filename = 'model_kidney.sav'
classifier = joblib.load(filename)


# for preddiction
def prediction(x):
    result = classifier.predict_proba(np.asarray(x).reshape(1,-1))
    if result[0][0] > 0.5:
        return "congrats our model say he is {}% sure you do not have kidney disease".format(int(result[0][0]*100))
    elif result[0][1]>0.5:
        return "sorry to say but our model say there is {} % chance you have a kidney disease".format(int(result[0][1]*100))
    else:
        return " sorry to say our model is not sure about you,its 50-50 condition"
value = [41.0,70.0,1.02,0.0,0.0,0,0,0,0,125.0,38.0,0.6,140.0,5.0,16.8,41,6300,5.9,0,0,0,0,0,0]
print(prediction(value))