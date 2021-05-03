# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:50:28 2020

@author: ladduu
"""


import pandas as pd
import joblib

# Loading the dataset
df = pd.read_csv('parkinsons.csv')

X = df.drop(columns=['name','status'])
y = df.iloc[:,17]

#checking count
from collections import Counter
print(Counter(y))

from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler(random_state=42)
Xn,yn = os.fit_resample(X, y)
print(Counter(yn))

# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating Random Forest Model and hyperparametertuning
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

'''
#making best model using Randomizedsearchedcv
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

est = RandomForestClassifier(n_jobs=-1)
rf_p_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,3),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,4),
              }
def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=9)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X, y)

'''

claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=100,bootstrap= False,criterion='gini',max_depth=None,max_features=1,min_samples_leaf= 1)


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score=accuracy_score(y_test,y_pred)
print(accuracy_score)

# Creating a pickle file for the classifier
filename = 'model_parkinsons.sav'
joblib.dump(classifier,filename)
