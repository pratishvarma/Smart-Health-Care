# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:02:37 2020

@author: ladduu
"""

# Importing essential libraries
import numpy as np
import pandas as pd


# Loading the dataset
df = pd.read_csv('kidney_disease.txt')

df.isnull().sum()

df.describe()

df.fillna(df.mean(),inplace = True)

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})

df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)


df['pe'] = df['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)

df["pcv"]=df["pcv"].fillna(method="ffill")
df["hemo"]=df["hemo"].fillna(method="ffill")
df=df.fillna(method="ffill")
df.dropna(inplace=True)
df=df.replace("\t?",31)

print(df.columns)
print(df.shape[1])

df.isnull().sum()

df.to_csv('Kidney.csv')