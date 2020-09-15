# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:58:48 2020

@author: sami
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import clust_csvread as ccr
from sklearn.metrics import silhouette_score

df=ccr.csv_read_function()
cor = df.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = True)

def rgb_to_hex(rgb):
  return '#%s' % ''.join(('%02x' % p for p in rgb))

df_ne= pd.read_csv('bank.csv')

sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(df_ne[['age', 'balance','day','duration','campaign','pdays','previous']])
    sse_.append([k, silhouette_score(df_ne[['age', 'balance','day','duration','campaign','pdays','previous']], kmeans.labels_)])

ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(df_ne[['age', 'balance','day','duration','campaign','pdays','previous']])
    ssd.append(model_clus.inertia_)

plt.plot(ssd)

kmeans = KMeans(n_clusters=3, random_state=0).fit(df_ne[['age', 'balance','day','duration','campaign','pdays','previous']])
df_ne['cluster'] = kmeans.labels_
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['age', 'balance','day','duration','campaign','pdays','previous'])
y_kmeans = kmeans.predict(df_ne[['age', 'balance','day','duration','campaign','pdays','previous']])
# plt.scatter(df_ne['campaign'], df_ne['pdays'], c=y_kmeans, s=50, cmap='viridis')

# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.figure(figsize=(7,7))
## Plot scatter by cluster / color, and centroids
# colors = [map(int, c) for c in kmeans.cluster_centers_]
# colors= list(map(rgb_to_hex, colors))
plt.figure(figsize=(7,7))
## Plot scatter by cluster / color, and centroids
colors = ["red", "green","blue"]
df_ne['color'] = df_ne['cluster'].map(lambda p: colors[p])
ax = df_ne.plot(    
    kind="scatter", 
    x="balance", y="age",
    figsize=(10,8),
    c = df_ne['color']
)
centroids.plot(
    kind="scatter", 
    x="balance", y="age", 
    marker="*", c=colors, s=550,
    ax=ax
)


sns.violinplot('age','balance',data=df,palette='coolwarm')
sns.countplot(x='age',hue='marital',data=df)

g = sns.FacetGrid(col='job',hue='housing',data=df,legend_out=False)
g.map(sns.scatterplot,'education','loan')


#starting consider the categorical features in K means

sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']])
    sse_.append([k, silhouette_score(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']], kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);

# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']])
    ssd.append(model_clus.inertia_)

plt.plot(ssd)

kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']])
df['cluster'] = kmeans.labels_
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['job', 'marital','education','default','housing','loan','contact', 'month','poutcome'])
y_kmeans = kmeans.predict(df[['job', 'marital','education','default','housing','loan','contact', 'month','poutcome']])

# plt.scatter(df_ne['campaign'], df_ne['pdays'], c=y_kmeans, s=50, cmap='viridis')

# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.figure(figsize=(7,7))
## Plot scatter by cluster / color, and centroids
# colors = [map(int, c) for c in kmeans.cluster_centers_]
# colors= list(map(rgb_to_hex, colors))
plt.figure(figsize=(7,7))
## Plot scatter by cluster / color, and centroids
colors = ["red", "green","blue"]
df['color'] = df['cluster'].map(lambda p: colors[p])
ax = df.plot(    
    kind="scatter", 
    x="poutcome", y="contact",
    figsize=(10,8),
    c = df['color']
)
centroids.plot(
    kind="scatter", 
    x="poutcome", y="contact", 
    marker="*", c=colors, s=550,
    ax=ax
)

#mixed data on K means
df=ccr.csv_read_function()
sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
    sse_.append([k, silhouette_score(df, kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);

# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(df)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)

model_clus5 = KMeans(n_clusters = 3, max_iter=50)
model_clus5.fit(df)

df.index = pd.RangeIndex(len(df.index))
df_km = pd.concat([df, pd.Series(model_clus5.labels_)], axis=1)
df_km.columns = ['age', 'job','marital', 'education','default','balance', 'housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y', 'ClusterID']
km_clusters_age = pd.DataFrame(df_km.groupby(["ClusterID"]).age.mean())
km_clusters_balance = pd.DataFrame(df_km.groupby(["ClusterID"]).balance.mean())

df1 = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_age,km_clusters_balance], axis=1)
df1.columns = ["ClusterID", "Age_mean","Balance_mean"]
df1.head()

sns.barplot(x=df1.ClusterID, y=df1.Age_mean)

sns.barplot(x=df1.ClusterID, y=df1.Balance_mean)





