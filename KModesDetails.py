# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:19:48 2020

@author: sami
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
import warnings
warnings.filterwarnings("ignore")

bank = pd.read_csv('bank.csv')
print(bank.head())

print(bank.columns)

bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day','poutcome']]


bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                              labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])

print(bank_cust.head())
bank_cust  = bank_cust.drop('age',axis = 1)
print(bank_cust.head())
print(bank_cust.info())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bank_cust = bank_cust.apply(le.fit_transform)
print(bank_cust.head())



job_df = pd.DataFrame(bank_cust['job'].value_counts())
sns.barplot(x=job_df.index, y=job_df['job'])

age_df = pd.DataFrame(bank_cust['age_bin'].value_counts())
sns.barplot(x=age_df.index, y=age_df['age_bin'])

#using K Mode with Kao initialization
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)

clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = bank_cust.columns

#using K Mode with Huang initialization
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(bank_cust)

cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(bank_cust)
    cost.append(kmode.cost_)
    
y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)

km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


bank_cust = bank_cust.reset_index()
clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

# Data for Cluster1
cluster1 = combinedDf[combinedDf.cluster_predicted==1]

cluster0 = combinedDf[combinedDf.cluster_predicted==0]

print(cluster1.info())
print(cluster0.info())

job1_df = pd.DataFrame(cluster1['job'].value_counts())
job0_df = pd.DataFrame(cluster0['job'].value_counts())

age1_df = pd.DataFrame(cluster1['age_bin'].value_counts())
age0_df = pd.DataFrame(cluster0['age_bin'].value_counts())


fig, ax =plt.subplots(1,2,figsize=(20,5))
sns.barplot(x=age1_df.index, y=age1_df['age_bin'], ax=ax[0])
sns.barplot(x=age0_df.index, y=age0_df['age_bin'], ax=ax[1])
fig.show()

print(cluster1['marital'].value_counts())
print(cluster0['marital'].value_counts())

print(cluster1['education'].value_counts())
print(cluster0['education'].value_counts())

bank_proto = bank[['job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','poutcome','age','duration']]
bank_proto.isnull().values.any()
bank.duration.mean()

columns_to_normalize  = ['age','duration']
columns_to_label = ['job', 'marital', 'education', 'default','housing', 'loan','contact','month','poutcome']

bank_proto[columns_to_normalize] = bank_proto[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))
le = preprocessing.LabelEncoder()
bank_proto[columns_to_label] = bank_proto[columns_to_label].apply(le.fit_transform)

print(bank_proto.head())
bank_proto_matrix = bank_proto.values

from kmodes.kprototypes import KPrototypes
# Running K-Prototype clustering
kproto = KPrototypes(n_clusters=5, init='Cao')
clusters = kproto.fit_predict(bank_proto_matrix, categorical=[0,1,2,3,4,5,6,7,8])

bank_proto['clusterID'] = clusters
print(kproto.cost_)

#Choosing optimal K
cost = []
for num_clusters in list(range(1,8)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
    kproto.fit_predict(bank_proto_matrix, categorical=[0,1,2,3,4,5,6,7,8])
    cost.append(kproto.cost_)
    
plt.plot(cost)

bank_protodf = pd.DataFrame(bank_proto['clusterID'].value_counts())

sns.barplot(x=bank_protodf.index, y=bank_protodf['clusterID'])