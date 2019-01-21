# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:59:01 2019

@author: mwon579
"""
import pandas as pd
import numpy as np

import random

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import pairwise_distances_argmin_min

'''
Read Data
'''
filename = 'Medicare_Provider_Util_Payment_PUF_CY2016.txt'

# n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
# s = 50000 #desired sample size
# skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

# data = pd.read_csv(filename, sep='\t', skiprows=skip)

data = pd.read_csv(filename, sep='\t')
colorado = data.loc[data['NPPES_PROVIDER_STATE'] == 'CO']
data = colorado


# drop useless columns
data.drop(data.columns[[0,1,2,3,4,5,7,8,12]], axis=1, inplace=True)
data.drop(['HCPCS_DESCRIPTION'], axis =1, inplace = True)
data.drop(['HCPCS_CODE'], axis =1, inplace = True)
data.drop(['PROVIDER_TYPE'], axis =1, inplace = True)
data.drop(['NPPES_PROVIDER_STATE'], axis=1, inplace = True)
data.drop(['NPPES_ENTITY_CODE'], axis=1, inplace = True)
data.drop(['MEDICARE_PARTICIPATION_INDICATOR'], axis=1, inplace = True)
data.drop(['PLACE_OF_SERVICE'], axis=1, inplace = True)
data.drop(['HCPCS_DRUG_INDICATOR'], axis=1, inplace = True)
data.drop(['LINE_SRVC_CNT'], axis=1, inplace = True)
data.drop(['BENE_UNIQUE_CNT'], axis=1, inplace = True)
data.drop(['BENE_DAY_SRVC_CNT'], axis=1, inplace = True)

'''
Convert all non numeric to string
'''

data_cat = data.select_dtypes(exclude=['float64', 'int64'])
cat_attribs = list(data_cat)

# take first 3 digits of zip
data[cat_attribs] = data[cat_attribs].astype(str)
data['NPPES_PROVIDER_ZIP'] = data['NPPES_PROVIDER_ZIP'].str[:3]


'''
Observe how the data is distributed and if there are any anomalies
'''
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(20,15))
plt.show()


'''
Top correlated pairs
'''
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

data_num = data.select_dtypes(include=['float64', 'int64'])

print(get_top_abs_correlations(data_num, n=10))

'''
Scatter Plot
'''
from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(12,8))


'''
Remove Redundant Variables and Log Transform data since it seems to be strongly skewed
'''
data.drop(['AVERAGE_MEDICARE_PAYMENT_AMT', 'AVERAGE_MEDICARE_STANDARD_AMT'], axis =1, inplace = True)

num_attribs = list(data.select_dtypes(include=['float64', 'int64']).columns)

data[num_attribs] = np.log(data[num_attribs])

data.hist(bins=50, figsize=(20,15))
plt.show()


'''
Prepare data for modeling
'''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_pipeline = SimpleImputer(strategy="mean")

encoder = OneHotEncoder(sparse=False)

full_pipeline = ColumnTransformer([
    ('num',num_pipeline, num_attribs),
    ('cat',encoder, cat_attribs),  
])

data_prepared = full_pipeline.fit_transform(data)

normalizer = StandardScaler()

filename = 'normalizer.sav'
joblib.dump(normalizer, filename)

data_prepared = normalizer.fit_transform(data_prepared)


onehotcategories = []

for i in range(len(full_pipeline.named_transformers_['cat'].categories_)):
    onehotcategories = onehotcategories + list(full_pipeline.named_transformers_['cat'].categories_[i])

data_prepared = pd.DataFrame(data_prepared, columns = num_attribs + onehotcategories)

filename = 'data_prepared.sav'
joblib.dump(data_prepared, filename)


'''
Clustering
'''
from sklearn.cluster import KMeans
for n in [3,4,5]:
    kmeans = KMeans(n_clusters = n).fit(data_prepared)
    
    from sklearn.externals import joblib
    # save the model to disk
    filename = 'finalized_model' + str(n) + '.sav'
    joblib.dump(kmeans, filename)
    
    len(kmeans.cluster_centers_)
    
    cluster_labels = kmeans.labels_
    n_clusters = len(kmeans.cluster_centers_)
    
    sc = plt.scatter(data['NPPES_PROVIDER_ZIP'], data['AVERAGE_MEDICARE_ALLOWED_AMT'], c=cluster_labels, s=10, cmap='viridis')
    plt.colorbar(sc)
    plt.show()
    
    
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data_prepared) + (n_clusters + 1) * 10])
    
    
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data_prepared, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data_prepared, cluster_labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()
    
loaded_model = joblib.load('finalized_model3.sav')
loaded_model
data_prepared['labels'] = loaded_model.labels_
cluster0 = data_prepared.loc[data_prepared['labels'] == 0]
cluster1 = data_prepared.loc[data_prepared['labels'] == 1]
cluster2 = data_prepared.loc[data_prepared['labels'] == 2]
cluster0.drop(['labels'], axis=1, inplace = True)
cluster0 = normalizer.inverse_transform(cluster0)
cluster0 = pd.DataFrame(cluster0, columns = num_attribs + onehotcategories)
cluster0[num_attribs] = np.exp(cluster0[num_attribs])
print(cluster0['AVERAGE_MEDICARE_ALLOWED_AMT'].sum())
cluster1.drop(['labels'], axis=1, inplace = True)
cluster1 = normalizer.inverse_transform(cluster1)
cluster1 = pd.DataFrame(cluster1, columns = num_attribs + onehotcategories)
cluster1[num_attribs] = np.exp(cluster1[num_attribs])
print(cluster1['AVERAGE_MEDICARE_ALLOWED_AMT'].sum())
cluster2.drop(['labels'], axis=1, inplace = True)
cluster2 = normalizer.inverse_transform(cluster2)
cluster2 = pd.DataFrame(cluster2, columns = num_attribs + onehotcategories)
cluster2[num_attribs] = np.exp(cluster2[num_attribs])
print(cluster2['AVERAGE_MEDICARE_ALLOWED_AMT'].sum())

'''
Find the instance that is closest to centroid 0
'''
## Returns array of the indices that are the closest to their respective clusters
closest, _ = pairwise_distances_argmin_min(loaded_model.cluster_centers_, data_prepared)

# closest[0] return the index in the dataset that is closest to center of cluser 0
closest0 = normalizer.inverse_transform(data_prepared.values[closest[0]])
closest0.shape = (1,258)
closest0 = pd.DataFrame(closest0, columns = num_attribs + onehotcategories)
closest0[closest0.columns[2:2+232]].idxmax(axis=1)
closest0[closest0.columns[2+233:-1]].idxmax(axis=1)