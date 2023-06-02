#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


census_data= pd.read_csv(r'C:\Users\Diwakar\OneDrive\Desktop\MS DS\MS_IIT_IU\Projects\Census_Income_Projects\Census-Income-project-Extra-Data-sets\census-income.csv')


# In[3]:


census_data.head()


# In[4]:


census_data.columns


# In[5]:


census_data


# In[6]:


census_data=census_data.rename(columns={' ': 'AnnualIncome'})


# In[7]:


census_data.columns


# In[8]:


census_data.isnull().sum()


# In[9]:


census_data.columns= census_data.columns.str.replace(" ", "")


# In[10]:


census_data.columns


# # Data Preprocessing

# In[11]:


#a.Replace missing values with NA
census_data= census_data.replace('?', np.nan)


# In[12]:


#b. Remove all rows with na values
census_data=census_data.dropna(axis=0)


# # Data Manipulation

# In[13]:


census_data.isna().sum()


# In[14]:


#a.Extract education and store it in census_ed
census_ed= census_data['education']


# In[15]:


#b.Extract columns from age to relationalship and store in census_seq
census_seq= census_data.loc[:, 'age':'relationship']
census_seq.head()


# In[16]:


#c.Extract the column number “5”, “8”, “11” and store it in “census_col”.
census_col=  census_data.iloc[:, [5, 8, 11]]
census_col.head()


# In[17]:


census_data['workclass'].value_counts()


# In[18]:


#d. Extract all the male employees who work in state-gov and store it in “male_gov”.
male_gov= census_data[(census_data['sex']=='Male') & (census_data['workclass']=='State-gov')]


# In[19]:


#e.Extract all the 39 year olds who either have a bachelor's degree or who are native of the United States and store the result in “census_us”.
census_us= census_data[(census_data['age']==39) | (census_data['education']=='Bachelors') | (census_data['native-country']=='United-States')]


# In[20]:


#f. Extract 200 random rows from the “census” data frame and store it in “census_200”
census_200= census_data.sample(200)


# In[21]:


#g. Get the count of different levels of the “workclass” column.
census_data['workclass'].value_counts()


# In[22]:


#h.Calculate the mean of the “capital.gain” column grouped according to “workclass”.
census_data.groupby('workclass')['capital-gain'].mean()


# In[23]:


#i. Create a separate dataframe with the details of males and females from the census data that has income more than 50,000.
census_inc_more_than_50K=census_data['AnnualIncome']> '>50K'


# In[24]:


#j. Calculate the percentage of people from the United States who are private employees and earn less than 50,000 annually
perc=census_data[(census_data['native-country']=='United-States') & (census_data['workclass']=='Private') & (census_data['AnnualIncome']> '>50K')]


# In[25]:


census_data['marital-status'].value_counts()


# In[26]:


#k. Calculate the percentage of married people in the census data.
census_data['marital-status'].value_counts()
marreid= 14976+418+23
marreid


# In[27]:


total= 14976+418+23+10683+4443
total


# In[28]:


married= 15417/30543
married


# In[29]:


# l. Calculate the percentage of high school graduates earning more than 50,000 annual
census_data[(census_data['education']=='HS-grad') & (census_data['AnnualIncome']=='>50K')]
higg_school_grads= 49/10501
higg_school_grads


# In[30]:


census_data['education'].value_counts()


# # 3.Linear Regrerssion

# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[34]:


lin_mod= LinearRegression()


# In[35]:


x= census_data['education-num']


# In[36]:


x=pd.DataFrame(x)


# In[37]:


y= census_data['hours-per-week']


# In[38]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=0)


# In[39]:


lin_mod.fit(x_train, y_train)


# In[40]:


y_pred=lin_mod.predict(x_test)


# In[41]:


y_pred


# In[43]:


err=mean_squared_error(y_test,y_pred)


# In[44]:


err


# # 4. Logistics Regression

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


log_reg= LogisticRegression()


# In[53]:


x1= census_data['occupation']
x1=pd.DataFrame(x1)


# In[54]:


y1= census_data['AnnualIncome']


# In[57]:


L=LabelEncoder()
x1=L.fit_transform(x1)
x1=pd.DataFrame(x1)


# In[58]:


x1_train, x1_test, y1_train, y1_test= train_test_split(x1,y1, test_size=0.35, random_state=0)


# In[59]:


log_reg.fit(x1_train, y1_train)


# In[60]:


y1_pred= log_reg.predict(x1_test)


# In[61]:


y1_pred


# In[62]:


acc=accuracy_score(y1_test, y1_pred)


# In[63]:


acc


# In[64]:


confusion_matrix(y1_test, y1_pred)


# # 4.b Multiple Logistics Regression

# In[65]:


x2= census_data[['age','workclass']]
x2=pd.DataFrame(x1)


# In[66]:


y2= census_data['AnnualIncome']


# In[67]:


x2_train, x2_test, y2_train, y2_test= train_test_split(x2,y2, test_size=0.20, random_state=0)


# In[68]:


log_reg.fit(x2_train, y2_train)


# In[69]:


y2_pred= log_reg.predict(x2_test)
y2_pred


# In[70]:


confusion_matrix(y2_test, y2_pred)


# In[71]:


acc1=accuracy_score(y2_test, y2_pred)
acc1


# # 5. Decision Tree

# In[72]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


census_data.education= census_data.education.replace([''])


# In[87]:


#census_data['workclass']= L.fit_transform(census_data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']])


# In[88]:


census_data['workclass']=L.fit_transform(census_data['workclass'])
census_data['education']=L.fit_transform(census_data['education'])
census_data['marital-status']=L.fit_transform(census_data['marital-status'])
census_data['occupation']=L.fit_transform(census_data['occupation'])
census_data['relationship']=L.fit_transform(census_data['relationship'])
census_data['race']=L.fit_transform(census_data['race'])
census_data['sex']=L.fit_transform(census_data['sex'])
census_data['native-country']=L.fit_transform(census_data['native-country'])


# In[89]:


x3= census_data.iloc[:, :-1]
x3=pd.DataFrame(x3)
y3= census_data['AnnualIncome']


# In[90]:


x3_train, x3_test, y3_train, y3_test= train_test_split(x3,y3, test_size=0.30, random_state=0)


# In[91]:


dec_tree= DecisionTreeClassifier()


# In[92]:


census_data.info()


# In[93]:


dec_tree.fit(x3_train, y3_train)


# In[94]:


y3_pred= dec_tree.predict(x3_test)


# In[95]:


y3_pred


# In[96]:


confusion_matrix(y3_test, y3_pred)


# In[97]:


acc2=accuracy_score(y3_test, y3_pred)


# In[98]:


acc2


# # 6. Random Forest

# In[100]:


from sklearn.ensemble import RandomForestClassifier


# In[109]:


rf=RandomForestClassifier(n_estimators=300)


# In[110]:


x4= census_data.iloc[:, :-1]
x4=pd.DataFrame(x4)
y4= census_data['AnnualIncome']


# In[111]:


x4_train, x4_test, y4_train, y4_test= train_test_split(x4,y4, test_size=0.30, random_state=0)


# In[112]:


rf.fit(x4_train, y4_train)


# In[113]:


y4_pred= rf.predict(x4_test)


# In[114]:


y4_pred


# In[115]:


confusion_matrix(y4_test, y4_pred)


# In[116]:


acc3=accuracy_score(y4_test, y4_pred)
acc3


# # 7. EDA

# In[118]:


popl=pd.read_csv(r'C:\Users\Diwakar\OneDrive\Desktop\MS DS\MS_IIT_IU\Projects\Census_Income_Projects\Census-Income-project-Extra-Data-sets\popdata.csv')


# In[120]:


popl.head()


# In[121]:


import matplotlib.pyplot as plt


# In[122]:


popl.info()


# In[123]:


#change date(object) to date format
popl['date']= pd.to_datetime(popl['date'])


# In[124]:


popl.info()


# In[127]:


popl.index


# In[128]:


popl.plot()


# In[177]:


popl.index= popl['date']


# In[178]:


from statsmodels.tsa.stattools import adfuller


# In[179]:


result=adfuller(popl['value'])


# In[180]:


print('p-value: %f' % result[1])


# In[181]:


if result[1]>0.05:
    print('Non-Stationary')
else:
    print('stationary')


# In[182]:


rolling_mean= popl.rolling(window=12).mean()
rolling_mean


# In[187]:


rolling_mean_detrended= popl- rolling_mean
rolling_mean_detrended= pd.DataFrame(rolling_mean_detrended)
ax1= plt.subplot(121)
rolling_mean_detrended.plot(figsize=(12,4), color="tab:red", title="Diff with rolling mean", ax=ax1)


# In[188]:


ax2= plt.subplot(122)
popl.plot(figsize=(12,4), color= "tab:red", title='original value', ax=ax2)


# In[189]:


#1. checking trends & Seasonality
from statsmodels.tsa.seasonal import seasonal_decompose


# In[230]:


seasonal_decompose(rolling_mean_detrended.dropna(), period= None )


# In[195]:


rolling_mean_detrended_diff= rolling_mean_detrended-rolling_mean_detrended.shift()


# In[196]:


rolling_mean_detrended_diff= rolling_mean_detrended_diff.dropna()


# In[201]:


get_ipython().system('pip install pmdarima')


# In[204]:


from pmdarima import auto_arima


# In[205]:


order= auto_arima(rolling_mean_detrended_diff['date'], trace= True)
order.summary()


# In[206]:


from statsmodels.tsa.arima_model import ARIMA


# In[215]:


#import statsmodels.api as sm


# In[212]:


train= rolling_mean_detrended_diff.iloc[:120]['value']
test= rolling_mean_detrended_diff.iloc[121:]['value']


# In[218]:


model= sm.tsa.arima.ARIMA(train, order=(3,0,3))
model_fit= model.fit()
model_fit.summary()


# In[220]:


rolling_mean_detrended_diff['predict']= model_fit.predict(start= len(train), end=len(train)+ len(test)-1, dynamic= True)


# In[222]:


rolling_mean_detrended_diff[['value', 'predict']].plot()


# In[224]:


from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


# In[226]:


#2. Predicting value for 6 Months
model= SARIMAX(train, order=(1,0,2), seasonal_order=(1,0,2,6))
model=model.fit


# In[227]:


rolling_mean_detrended_diff['predict']= model.predict(start= len(train), end=len(train)+ len(test)-1, dynamic= True)


# In[229]:


rolling_mean_detrended_diff[['value', 'predict']].plot()


# In[ ]:




