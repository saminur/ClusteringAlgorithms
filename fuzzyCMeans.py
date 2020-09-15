# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:28:40 2020

@author: sami
"""

import numpy as np
import random
import operator
import math
import clust_csvread as ccr
import seaborn as sns
df= ccr.csv_read_function()

columns = list(df.columns)
features = columns[:len(columns)-1]
class_labels = list(df[columns[-1]])
df = df[features]

# Number of Attributes
num_attr = len(df.columns) - 1

# Number of Clusters
k = 5

# Maximum number of iterations
MAX_ITER = 100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 2.00

def accuracy(cluster_labels, class_labels):
    county = [0,0]
    countn = [0,0]
    tp = [0, 0]
    tn = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    
    for i in range(len(df)):
        # Yes = 1, No = 0
        if cluster_labels[i] == 1 and class_labels[i] == 1:
            tp[0] = tp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 0:
            tn[0] = tn[0] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 0:
            fp[0] = fp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 1:
            fn[0] = fn[0] + 1
    
    for i in range(len(df)):
        # Yes = 0, No = 1
        if cluster_labels[i] == 0 and class_labels[i] == 1:
            tp[1] = tp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 0:
            tn[1] = tn[1] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 0:
            fp[1] = fp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 1:
            fn[1] = fn[1] + 1
    
    a0 = ((tp[0] + tn[0]))*1.0/(tp[0] + tn[0] + fn[0] + fp[0])
    a1 = ((tp[1] + tn[1]))*1.0/(tp[1] + tn[1] + fn[1] + fp[1])
    p0 = (tp[0])*1.0/(tp[0] + fp[0])
    p1 = (tp[1])*1.0/(tp[1] + fp[1])
    r0 = (tp[0])*1.0/(tp[0] + fn[0])
    r1 = (tp[1])*1.0/(tp[1] + fn[1])
    
    accuracy = [a0*100,a1*100]
    precision = [p0*100,p1*100]
    recall = [r0*100,r1*100]
    
    return accuracy, precision, recall


def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1
    print(membership_mat)
    return cluster_labels, cluster_centers


labels, centers = fuzzyCMeansClustering()


a,p,r = accuracy(labels, class_labels)


import matplotlib.pyplot as plt
from seaborn import scatterplot as scatter

# sns.set_style("white")
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 20
# plt.rcParams['xtick.labelsize'] = 15
# plt.rcParams['ytick.labelsize'] = 15
# plt.rcParams['legend.fontsize'] = 15
# plt.rcParams['figure.titlesize'] = 20
# plt.rcParams['figure.figsize'] = (8,7)

# plt.scatter(centers[:, 0], centers[:, 1], c=labels, marker='x')    
# plt.title('Data points and cluster centroids')
# plt.show()
print("Accuracy = " + str(a))
print("Precision = " + str(p))
print("Recall = " + str(r))


f, axes = plt.subplots(1, 2, figsize=(11,5))
scatter(df.to_numpy()[:,0], df.to_numpy()[:,1], ax=axes[0])
scatter(df.to_numpy()[:,0], df.to_numpy()[:,1], ax=axes[1], hue=labels)
scatter(centers[:,0], centers[:,1], ax=axes[1],marker="s",s=200)
plt.show()