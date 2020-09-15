# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:56:54 2020

@author: sami
"""


import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import clust_csvread as ccr
import seaborn as sns
df=ccr.csv_read_function()
#[['age', 'balance','day','duration','campaign','pdays','previous']
 #[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']
hierarchical_linkage = linkage(df, method='ward')

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')

dendrogram(
    hierarchical_linkage,
    show_leaf_counts=True
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
df=ccr.csv_read_function()
cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
cluster.fit_predict(df)
df=df.values.tolist()
df['clusterID'] = cluster.labels_

bank_df = pd.DataFrame(df['clusterID'].value_counts())

sns.barplot(x=bank_df.index, y=bank_df['clusterID'])
