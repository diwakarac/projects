#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


walmart=pd.read_csv(r'C:\Users\Diwakar\OneDrive\Desktop\MS DS\MS_IIT_IU\Projects\Capstone_Project\Capstone-Dataset\Walmart.csv')


# In[3]:


walmart.head()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


x=walmart['Store']
y= walmart['Weekly_Sales']


# In[6]:


plt.scatter(x,y, c=walmart['CPI'])
plt.show()


# In[7]:


walmart.info()


# In[8]:


walmart.isnull().sum()


# In[9]:


walmart.columns


# In[10]:


def scatter(walmart, column):
    plt.figure()
    plt.scatter(walmart[column] , walmart['Weekly_Sales'])
    plt.ylabel('Weekly_Sales')
    plt.xlabel(column)

scatter(walmart, 'Fuel_Price')  # with respect to Fuel_Price
scatter(walmart, 'Date')  # with respect to date
scatter(walmart, 'CPI')  # with respect to CPI
scatter(walmart, 'Holiday_Flag') # with respect to Holiday
scatter(walmart, 'Unemployment')  # with respect to Unemployment
scatter(walmart, 'Temperature') # with respect to Temperature
scatter(walmart, 'Store') # with respect to Store


# In[11]:


walmart.corr()


# In[12]:


walmart.describe()


# In[13]:


walmart.max()


# In[14]:


Sales_groupby = walmart.groupby('Store')['Weekly_Sales'].sum()
print("Store Number {} has maximum Sales. Sum of Total Sales {}".
format(Sales_groupby.idxmax(),Sales_groupby.max()))


# In[15]:


maxstd=pd.DataFrame(walmart.groupby('Store').agg({'Weekly_Sales':['std','mean']}))
maxstd = maxstd.reset_index()
maxstd['CoV'] =(maxstd[('Weekly_Sales','std')]/maxstd[('Weekly_Sales','mean')])*100
maxstd.loc[maxstd[('Weekly_Sales','std')]==maxstd[('Weekly_Sales','std')].max()]


# In[16]:


Qrt_growth = walmart.groupby('Store').agg({'Weekly_Sales':['mean','std']})
Qrt_growth.head()


# In[17]:


data_Q32012 = walmart[(pd.to_datetime(walmart['Date']) >= pd.to_datetime('07-01-2012')) & (pd.to_datetime(walmart['Date']) <= pd.to_datetime('09-30-2012'))]
data_growth = data_Q32012.groupby(['Store'])['Weekly_Sales'].sum()
print("Store Number {} has Good Quartely Growth in Q3'2012 {}".format(data_growth.idxmax(),data_growth.max()))


# In[18]:


walmart_data = walmart.groupby('Holiday_Flag')['Weekly_Sales'].mean()
walmart['Date'] =  pd.to_datetime(walmart['Date'])
walmart["Day"]= pd.DatetimeIndex(walmart['Date']).day
walmart['Month'] = pd.DatetimeIndex(walmart['Date']).month
walmart['Year'] = pd.DatetimeIndex(walmart['Date']).year


# In[19]:


def plot_line(df,holiday_dates,holiday_label):
    fig, ax = plt.subplots(figsize = (15,5))  
    ax.plot(df['Date'],df['Weekly_Sales'],label=holiday_label)
    
    for day in holiday_dates:
        day = datetime.strptime(day, '%d-%m-%Y')
        plt.axvline(x=day, linestyle='--', c='r')
    

    plt.title(holiday_label)
    x_dates = df['Date'].dt.strftime('%Y-%m-%d').sort_values().unique()
    xfmt = dates.DateFormatter('%d-%m-%y')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(dates.DayLocator(1))
    plt.gcf().autofmt_xdate(rotation=90)
    plt.show()


# In[26]:


#The sales increased during thanksgiving. And the sales decreased during christmas.


# In[27]:


import seaborn as sns
# find outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = walmart[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(walmart[column], ax=axs[i])


# In[28]:


# drop the outliers     
without_outlier = walmart[(walmart['Unemployment']<10) & (walmart['Unemployment']>4.5) & (walmart['Temperature']>10)]
without_outlier


# In[29]:


# check outliers 
fig, axs = plt.subplots(4,figsize=(6,18))
X = without_outlier[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(without_outlier[column], ax=axs[i])


# In[30]:


# Import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[31]:


# Select features and target 
X = without_outlier[['Store','Fuel_Price','CPI','Unemployment']]
y = without_outlier['Weekly_Sales']

# Split data to train and test (0.80:0.20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[32]:


# Linear Regression model
print('Linear Regression:')
print()
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Accuracy:',reg.score(X_train, y_train)*100)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[33]:


# Random Forest Regressor
print('Random Forest Regressor:')
print()
rfr = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)        
rf_model=rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print('Accuracy:',rfr.score(X_test, y_test)*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


sns.scatterplot(y_pred, y_test);


# In[34]:


final_y_prediction = rf_model.predict(X_test)
final_y_prediction


# In[46]:


from statsmodels.tsa.arima.model import ARIMA

# convert the date column to a datetime type
walmart['Date'] = pd.to_datetime(walmart['Date'])

# group the data by store
store_groups = walmart.groupby('Store')

# create a dictionary to hold the forecasts
forecasts = {}

# loop through each store group and make a forecast
for name, group in store_groups:
    # set the date column as the index
    group = group.set_index('Date')
    # resample data to monthly frequency
    group = group.resample('M').sum()
    # Fit the ARIMA model
    model = ARIMA(group['Weekly_Sales'], order=(1,1,0))
    model_fit = model.fit()
    # Forecast for next 12 months
    forecast = model_fit.forecast(steps=12)[0]
    # add the forecast to the dictionary
    forecasts[name] = forecast

# view the forecasts for each store
for store, forecast in forecasts.items():
    print(f'Forecast for Store {store}:')
    print(forecast)


# In[45]:


get_ipython().system('pip install fbprophet')


# In[44]:


import pandas as pd
from fbprophet import Prophet

# load the sales data into a pandas dataframe
sales_data = pd.read_csv('walmart_sales.csv')

# convert the date column to a datetime type
sales_data['Date'] = pd.to_datetime(sales_data['date'])

# group the data by store
store_groups = sales_data.groupby('store')

# create a dictionary to hold the forecasts
forecasts = {}

# loop through each store group and make a forecast
for name, group in store_groups:
    # set the date column as the index
    group.set_index('date', inplace=True)
    # create a Prophet model
    model = Prophet()
    # fit the model to the store's sales data
    model.fit(group)
    # create a dataframe to hold the forecast
    future = model.make_future_dataframe(periods=12, freq='M')
    # make the forecast
    forecast = model.predict(future)
    # add the forecast to the dictionary
    forecasts[name] = forecast

# view the forecasts for each store
for store, forecast in forecasts.items():
    print(f'Forecast for Store {store}:')
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


# In[ ]:




