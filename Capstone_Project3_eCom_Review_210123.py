#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


ecom_review= pd.read_csv(r'C:\Users\Diwakar\OneDrive\Desktop\MS DS\MS_IIT_IU\Projects\Capstone_Project\Capstone-Dataset\Reviews.csv')


# In[3]:


ecom_review.head()


# In[4]:


ecom_review.describe()


# In[5]:


ecom_review.corr()


# In[6]:


ecom_review.max()


# In[7]:


ecom_review.isnull().sum()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(2,figsize=(6,18))
X = ecom_review[['Score']]
for i,column in enumerate(X):
    sns.boxplot(ecom_review[column], ax=axs[i], width=0.8)


# In[9]:


# Distribution of scores
plt.hist(ecom_review["Score"])
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Distribution of Scores")
plt.show()


# In[10]:


# Relationship between helpfulness numerator and denominator
plt.scatter(ecom_review["HelpfulnessNumerator"], ecom_review["HelpfulnessDenominator"])
plt.xlabel("Helpfulness Numerator")
plt.ylabel("Helpfulness Denominator")
plt.title("Helpfulness Numerator vs Denominator")
plt.show()


# In[11]:


# Number of reviews by date
ecom_review["Time"] = pd.to_datetime(ecom_review["Time"], unit='s')
reviews_by_date = ecom_review.groupby(ecom_review["Time"].dt.date).count()["Id"]
plt.plot(reviews_by_date)
plt.xlabel("Date")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews by Date")
plt.show()


# In[12]:


reviews_per_product = ecom_review.groupby("ProductId").size().reset_index(name="Reviews").sort_values(by="Reviews",ascending=False)
print(reviews_per_product.head())


# In[13]:


reviews_per_user = ecom_review.groupby("UserId").size().reset_index(name="Reviews").sort_values(by='Reviews', ascending=False)
print(reviews_per_user.head())


# In[14]:


average_score_per_product = ecom_review.groupby("ProductId")["Score"].mean().reset_index().sort_values(by='Score',ascending=False)
average_score_per_product.head()


# In[15]:


import nltk


# In[16]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[17]:


# Create an instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[18]:


# Add a new column to the dataframe with the sentiment scores
ecom_review["Sentiment"] = ecom_review["Text"].apply(lambda x: sia.polarity_scores(x)["compound"])


# In[19]:


# Plot the distribution of sentiment scores
plt.hist(ecom_review["Sentiment"])
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.title("Distribution of Sentiment Scores")
plt.show()


# In[21]:


#Classify
# Function to classify a review as positive, negative, or neutral
def classify_review(text):
    sentiment = sia.polarity_scores(text)["compound"]
    if sentiment > 0.5:
        return "Positive"
    elif sentiment < -0.5:
        return "Negative"
    else:
        return "Neutral"

# Add a new column to the dataframe with the sentiment labels
ecom_review["Sentiment_Label"] = ecom_review["Text"].apply(classify_review)

# Print the first few rows of the dataframe to see the new column
print(ecom_review.head())


# In[22]:


#Using text blob
from textblob import TextBlob

# Function to classify a review as positive, negative, or neutral
def classify_review(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Add a new column to the dataframe with the sentiment labels
ecom_review["Sentiment_Label"] = ecom_review["Text"].apply(classify_review)

# Print the first few rows of the dataframe to see the new column
print(ecom_review.head())


# In[ ]:




