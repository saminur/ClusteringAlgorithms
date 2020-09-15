# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:20:44 2020

@author: samin
"""

import pyclustering 

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np    
from pyclustering.cluster.kmedoids import kmedoids;
from pyclustering.utils import read_sample;
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.elbow import elbow
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
import clust_csvread as ccr
def gower_distance(X):
    
    individual_variable_distances = []
    epsilon = 10**(-8)
    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / max(np.ptp(feature.values), epsilon)
            
        individual_variable_distances.append(feature_dist)

    return np.array(individual_variable_distances).mean(0)
#[['age', 'balance','day','duration','campaign','pdays','previous']
df= pd.read_csv('bank.csv') #[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']
D = gower_distance(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']])
df_encode= ccr.csv_read_function()
# search_instance = silhouette_ksearch(df_encode, 2, 10, algorithm=silhouette_ksearch_type.KMEDOIDS).process()
# amount = search_instance.get_amount()
# scores = search_instance.get_scores()

initial_medoids = kmeans_plusplus_initializer(D, 4).initialize(return_index=True)
kmedoids_instance = kmedoids(D,initial_medoids, data_type='distance_matrix');

kmedoids_instance.process();
clusters = kmedoids_instance.get_clusters();
medoids = kmedoids_instance.get_medoids()
print(clusters)
# Calculate Silhouette score
score = silhouette(D, clusters).process().get_score()
dd= max(score)

#from pyclustering.samples.definitions import SIMPLE_SAMPLES
#sample = read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE3)
df_encode= ccr.csv_read_function()
df_encode=df_encode.values.tolist()

#for bar plot
import seaborn as sns
#get values from checking manually the size of each cluster
df_bank=pd.DataFrame({'clusterID':[8, 85, 3179, 1300,1855]})
sns.barplot(x=df_bank.index, y=df_bank['clusterID'])

# visualizer = cluster_visualizer_multidim()
# visualizer.append_clusters(clusters, df_encode)
# visualizer.show()

from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
# generating correlation data 
df_encode= ccr.csv_read_function()
df = df_encode.corr() 
df.index = range(0, len(df)) 
df.rename(columns = dict(zip(df.columns, df.index)), inplace = True) 
df = df.astype(object) 
  
''' Generating coordinates with  
corresponding correlation values '''
for i in range(0, len(df)): 
    for j in range(0, len(df)): 
        if i != j: 
            df.iloc[i, j] = (i, j, df.iloc[i, j]) 
        else : 
            df.iloc[i, j] = (i, j, 0) 
  
df_list = [] 
  
# flattening dataframe values 
for sub_list in df.values: 
    df_list.extend(sub_list) 
  
# converting list of tuples into trivariate dataframe 
plot_df = pd.DataFrame(df_list) 
  
fig = plt.figure() 
ax = Axes3D(fig) 
  
# plotting 3D trisurface plot 
ax.plot_trisurf(plot_df[0], plot_df[1], plot_df[2], cmap = cm.jet, linewidth = 0.2) 
  
plt.show() 


