# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:01:11 2020

@author: sami
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import scipy.spatial.distance as ssd
# import dimension_reduction as dr
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

df= pd.read_csv('bank.csv')

gower_dist=gower_distance(df)

# import gower
# gower_mat = gower.gower_matrix(gower_dist)
# hierarchal_cluster1 = linkage(gower_dist)
distArray = ssd.squareform(gower_dist)
hierarchal_cluster = linkage(distArray, method='ward')

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')

dendrogram(
    hierarchal_cluster,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=7,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True , # to get a distribution impression in truncated branches
    show_leaf_counts=True
)
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 5, linkage ='ward')
y_hc=hc.fit_predict(gower_dist)

plt.scatter([y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('Clusters of Customers (Hierarchical Clustering Model)')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()
# from scipy.cluster.hierarchy import fcluster
# from sklearn.metrics import silhouette_samples, silhouette_score

# silhouette_list = []
# no_of_clusters = [ 3, 4, 5, 6,7,8,9,10,11]
# for n_clusters in no_of_clusters:
#     nodes = fcluster(hierarchal_cluster, n_clusters, criterion="maxclust")
#     silhouette_avg = silhouette_score(gower_dist, nodes, metric="mahalanobis")
#     silhouette_list.append(silhouette_avg)

# plt.plot(no_of_clusters, silhouette_list)
# plt.show()


# X_embedded = TSNE(n_components=2).fit_transform(gower_dist)
# print(X_embedded.shape)

X_embedded = dr.fit(gower_dist)


# X_embedded = fit(gower_dist)
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# palette = sns.color_palette("bright", 2)

y=df.iloc[:,-1]
y=y.eq('yes').mul(1)

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 3)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, s=50, alpha=0.5);
plt.show()
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend = "full", palette=palette)

from sklearn.manifold import TSNE
tsne = TSNE()
X_embedded = tsne.fit_transform(gower_dist)
palette = sns.color_palette("bright", 12)
mapping = {'admin.':1, 'blue-collar':2, 'entrepreneur':3, 'housemaid':4, 'management':5, 'retired':6, 'self-employed':7, 'services':8, 'student':9, 'technician':10, 'unemployed':11, 'unknown':12}
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, s=50, alpha=0.5);

plt.show()
# y=df['job'] 
# y= y.replace(mapping)
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
# from yellowbrick.cluster import KElbowVisualizer
# visualizer = KElbowVisualizer(hierarchal_cluster, k=(2,11), metric='silhouette', timings=False)

# # Fit the data and visualize
# visualizer.fit(gower_dist)    
# visualizer.poof() 