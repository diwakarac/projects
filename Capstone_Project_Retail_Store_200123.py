#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


retail_store=pd.read_csv('C:/Users/Diwakar/OneDrive/Desktop/MS DS/MS_IIT_IU/Projects/Capstone_Project/Capstone-Dataset/OnlineRetail.csv', encoding='latin-1')


# In[5]:


retail_store.head()


# In[7]:


retail_store.describe()


# In[8]:


retail_store.corr()


# In[9]:


retail_store.max()


# In[13]:


retail_store.isnull().sum()


# In[14]:


135080/541909


# In[15]:


retail_store['total_purchase']=retail_store['Quantity']*retail_store['UnitPrice']
retail_store['total_purchase'].head()


# In[16]:


retail_store.head()


# In[21]:


popular_products = retail_store.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
print(popular_products)


# In[23]:


sales_by_country = retail_store.groupby("Country")["Quantity"].sum().sort_values(ascending=False)
print(sales_by_country)


# In[26]:


purchase_frequency = retail_store.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending=False)
average_spend = retail_store.groupby("CustomerID")["Quantity"].sum().sort_values(ascending=False)
print(purchase_frequency)
print(average_spend)


# In[29]:


retail_store['InvoiceDate'] = pd.to_datetime(retail_store['InvoiceDate'])
sales_by_month = retail_store.groupby(retail_store['InvoiceDate'].dt.strftime('%B'))['Quantity'].sum().sort_values(ascending=False)
print(sales_by_month)


# In[31]:


# stock forecasting
stock_forecast = retail_store.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
print(stock_forecast)


# In[41]:


import seaborn as sns
fig, axs = plt.subplots(2,figsize=(6,18))
X = retail_store[['UnitPrice']]
for i,column in enumerate(X):
    sns.boxplot(retail_store[column], ax=axs[i], width=0.8)


# In[42]:


without_outlier = retail_store[(retail_store['UnitPrice']>5000)]
without_outlier


# In[48]:


retail_store["segment"] = "low value"
retail_store.loc[retail_store["UnitPrice"] >= 50, "segment"] = "high value"
purchase_counts = retail_store.groupby("CustomerID")["UnitPrice"].count()
purchase_counts = purchase_counts.reset_index()
purchase_counts.columns = ["CustomerID", "purchase_counts"]
retail_store = pd.merge(retail_store, purchase_counts, on="CustomerID")
retail_store.loc[retail_store["purchase_counts"] >= 10, "segment"] = "frequent" + " " + retail_store["segment"]
retail_store.loc[retail_store["purchase_counts"] < 3, "segment"] = "infrequent" + " " + retail_store["segment"]
print(retail_store[["CustomerID", "segment"]])


# In[49]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# select relevant features for clustering
features = retail_store[["Quantity", "UnitPrice"]]

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# run k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_data)

# assign cluster labels to customers
retail_store["segment"] = kmeans.labels_

# print the segmented data
print(retail_store[["CustomerID", "segment"]])


# In[50]:


retail_store['segment'].value_counts()


# In[51]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# select relevant features for clustering
features = retail_store[["Quantity", "UnitPrice"]]

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# run hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(scaled_data)

# assign cluster labels to customers
retail_store["segment"] = agg_clustering.labels_

# print the segmented data
print(retail_store[["CustomerID", "segment"]])


# In[53]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create an empty list to store WCSS values
wcss = []

# Fit k-means for k in range 1 to 11
for k in range(3):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




