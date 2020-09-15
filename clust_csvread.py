# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:27:24 2020

@author: sami
"""
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder 
import random
import operator
import math

def csv_read_function():
    df= pd.read_csv('bank.csv')
    df.columns = df.columns.str.replace(' ', '')
    df.head()
    le = LabelEncoder().fit(df['job'])
    df['job'] = le.transform(df['job'])

    le1 = LabelEncoder().fit(df['marital'])
    df['marital'] = le1.transform(df['marital'])

    le2 = LabelEncoder().fit(df['education'])
    df['education'] = le2.transform(df['education'])

    le3 = LabelEncoder().fit(df['default'])
    df['default'] = le3.transform(df['default'])

    le4 = LabelEncoder().fit(df['housing'])
    df['housing'] = le4.transform(df['housing'])

    le5 = LabelEncoder().fit(df['loan'])
    df['loan'] = le5.transform(df['loan'])

    le6 = LabelEncoder().fit(df['contact'])
    df['contact'] = le6.transform(df['contact'])

    le7 = LabelEncoder().fit(df['month'])
    df['month'] = le7.transform(df['month'])

    le8 = LabelEncoder().fit(df['poutcome'])
    df['poutcome'] = le8.transform(df['poutcome'])

    le9 = LabelEncoder().fit(df['y'])
    df['y'] = le9.transform(df['y'])

    L = list(le.inverse_transform(df['job']))
    d = dict(zip(le.classes_, le.transform(le.classes_)))
    print (d)

    L1 = list(le1.inverse_transform(df['marital']))
    d1 = dict(zip(le1.classes_, le1.transform(le1.classes_)))
    print (d1)

    L2 = list(le2.inverse_transform(df['education']))
    d2 = dict(zip(le2.classes_, le2.transform(le2.classes_)))
    print (d2)

    L3 = list(le3.inverse_transform(df['default']))
    d3 = dict(zip(le3.classes_, le3.transform(le3.classes_)))
    print (d3)

    L4 = list(le4.inverse_transform(df['housing']))
    d4 = dict(zip(le4.classes_, le4.transform(le4.classes_)))
    print (d4)

    L5 = list(le5.inverse_transform(df['loan']))
    d5 = dict(zip(le5.classes_, le5.transform(le5.classes_)))
    print (d5)

    L6 = list(le6.inverse_transform(df['contact']))
    d6 = dict(zip(le6.classes_, le6.transform(le6.classes_)))
    print (d6)

    L7 = list(le7.inverse_transform(df['month']))
    d7 = dict(zip(le7.classes_, le7.transform(le7.classes_)))
    print (d7)

    L8 = list(le8.inverse_transform(df['poutcome']))
    d8 = dict(zip(le8.classes_, le8.transform(le8.classes_)))
    print (d8)

    L9 = list(le9.inverse_transform(df['y']))
    d9 = dict(zip(le9.classes_, le9.transform(le9.classes_)))
    print (d9)

    print(df.info())
    return df
