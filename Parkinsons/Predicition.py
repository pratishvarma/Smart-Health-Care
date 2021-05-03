# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:06:46 2021

@author: ladduu
"""

import numpy as np
import joblib

# opening a pickle file for the classifier
filename = 'model_parkinsons.sav'
classifier = joblib.load(filename)

# for preddiction
def prediction(x):
    
    result = classifier.predict_proba(np.asarray(x).reshape(1,-1))
    if result[0][0] > 0.5:
        return "congrats our model say he is {}% sure you dont have parkinsons".format(int(result[0][0]*100))
    elif result[0][1]>0.5:
        return "sorry to say but our model say there is {} % chance you have a parkinsons".format(int(result[0][1]*100))
    else:
        return " sorry to say our model is not sure about you,its 50-50 condition"
value = [116.676,137.871,111.366,0.00997,0.00009,0.00502,0.00698,0.01505,0.05492,0.517,0.02924,0.04005,0.03772,0.08771,0.01353,
         20.644,0.434969,0.819235,-4.117501,0.334147,2.405554,0.368975]
print(prediction(value))