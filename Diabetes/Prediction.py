# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:03:12 2021

@author: ladduu
"""

import numpy as np
import joblib

# opening a pickle file for the classifier
filename = 'model_diabetes.sav'
classifier = joblib.load(filename)


# for preddiction
def prediction(x):
    result = classifier.predict_proba(np.asarray(x).reshape(1,-1))
    if result[0][0] > 0.5:
        return "congrats our model say he is {}% sure you dont have diabetes".format(int(result[0][0]*100))
    elif result[0][1]>0.5:
        return "sorry to say but our model say there is {} % chance you have a diabetes".format(int(result[0][1]*100))
    else:
        return " sorry to say our model is not sure about you,its 50-50 condition"
value = [8,167,106,46,231,37.6,0.165,43]
print(prediction(value))