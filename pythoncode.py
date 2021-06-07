#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT ALL NEEDED LIBRARIES
import pandas as pd
from os import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from geopy.distance import geodesic
from geopy.distance import great_circle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#IMPORT CSV FILE AND CONVERT THE REQUIRED DATA TYPES
getcwd()
train_cab=pd.read_csv('C:\\Users\\My guest\\Desktop\\cab project\\train_cab.csv')
test_cab = pd.read_csv('C:\\Users\\My guest\\Desktop\\cab project\\test.csv')


# In[4]:


#GET THE FEEL OF THE DATA
train_cab.head()


# In[5]:


train_cab.describe()


# In[6]:


train_cab.info()


# In[7]:


#Converting data types
data=[train_cab,test_cab]
train_cab['fare_amount']  = pd.to_numeric(train_cab['fare_amount'],errors='coerce')
for i in data:
    i['pickup_datetime']  = pd.to_datetime(i['pickup_datetime'],errors='coerce')
  
    


# In[8]:


train_cab.info()


# In[9]:


test_cab.info()


# In[10]:


test_cab.describe()


# In[11]:


test_cab.head()


# In[12]:


test_cab.nunique()


# In[13]:


train_cab.nunique()


# # Graphical EDA - Data Visualization

# In[14]:


# setting up the sns for plots
sns.set(style='darkgrid',palette='Set2')


# In[15]:


#histogram plots
plt.figure(figsize=(20,20))
plt.subplot(321)
_ = sns.distplot(train_cab['fare_amount'],bins=50)
plt.subplot(322)
_ = sns.distplot(train_cab['pickup_longitude'],bins=50)
plt.subplot(323)
_ = sns.distplot(train_cab['pickup_latitude'],bins=50)
plt.subplot(324)
_ = sns.distplot(train_cab['dropoff_longitude'],bins=50)
plt.subplot(325)
_ = sns.distplot(train_cab['dropoff_latitude'],bins=50)
plt.subplot(326)
_ = sns.distplot(train_cab['passenger_count'],bins=50)
plt.savefig('hist.png')
plt.show()


# # missing value analysis in train_cab

# In[16]:


#create dataframe with missing value percentage
#counting np.nan
missing_val=pd.DataFrame(train_cab.isnull().sum())
#reset index
missing_val=missing_val.reset_index()
missing_val.to_csv('miss_val.csv',index=False)
missing_val


# In[17]:


#remane variable
missing_val=missing_val.rename(columns={'index':'variable',0:"missing_percent"})
#calculate percentage
missing_val['missing_percent']=(missing_val['missing_percent']/len(train_cab))*100
#descending order missing value also chane the index as regular 
missing_val=missing_val.sort_values('missing_percent',ascending=False).reset_index(drop=True)
#missijng_val save the file in hard disk
missing_val.to_csv('miss_per.csv',index=False)
missing_val


# ###Dealing with missing values in fare_amount

# In[18]:


# dealing with nas in 1 st column
#actual value=10
#mean imputation=15.015317000186862
#median imputation=8.5
#knn imputation=10.06867
train_cab.fare_amount[100]
train_cab.fare_amount[100]=np.nan
train_cab.fare_amount.mean()
train_cab.fare_amount.median()
#checking knn imputation value
cabk=train_cab.drop('pickup_datetime',axis=1)
cabk=pd.DataFrame(KNN(k=5).fit_transform(cabk),columns=cabk.columns)
cabk.fare_amount[100]


# In[19]:


sum(cabk.fare_amount!=0)


# In[20]:


cabk.loc[cabk.fare_amount<1]


# In[21]:


#knn imputation is closest to actual value so 
train_cab.fare_amount=cabk.fare_amount


# In[22]:


#check the imputed value
train_cab.fare_amount[100]


# In[23]:


#check missing values left any
train_cab.fare_amount.isna().sum()


# ###Dealing with missing values in pickup_datetime column

# In[24]:


#missing val analysis for 2nd column
train_cab.pickup_datetime.fillna(method='ffill',inplace=True)


# In[25]:


#check it
train_cab.pickup_datetime.isna().sum()


# ## there are no missing values in pickup_longitude ,pickup_latitude,dropoff_longitude,dropoff_latitude

#  ### dealing with np. nan  present in passenger_count

# In[26]:


# missing val analysis for last column
#actual value=1
#mean imputation=2.6251714446318153
#median imputation=1
#knn imputation= 1.2073240237942406
train_cab.passenger_count[100]
train_cab.passenger_count[100]=np.nan
train_cab.passenger_count.mean()
train_cab.passenger_count.median()
cabk=train_cab.drop('pickup_datetime',axis=1)
cabk=pd.DataFrame(KNN(k=5).fit_transform(cabk),columns=cabk.columns)
cabk.passenger_count[100]


# In[27]:


# so median is close to the actual value
# median value is closet amon all method for fare_amount value
train_cab['passenger_count']=train_cab['passenger_count'].fillna(train_cab['passenger_count'].median())


# In[28]:


train_cab.passenger_count[100]
train_cab.passenger_count.isna().sum()


# In[29]:


#check for missing values in entire data set
train_cab.isna().sum()


# ###### data train_cab is free from missing values

# # OUTLIER ANALYSIS

# In[30]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['fare_amount'],data=train_cab,orient='h')
plt.title('Boxplot of fare_amount')
plt.savefig('bp of fare_amount.png')
plt.show()


# In[31]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=test_cab['pickup_longitude'],data=train_cab,orient='v')
plt.title('Boxplot of test_pickup_longitude')
plt.savefig('bp of test_pickup_longitude.png')
plt.show()


# In[32]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=test_cab['pickup_latitude'],data=train_cab,orient='h')
plt.title('Boxplot of test_pickup_latitude')
plt.savefig('bp of test_pickup_latitude.png')
plt.show()


# In[33]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=test_cab['dropoff_longitude'],data=train_cab,orient='v')
plt.title('Boxplot of test_dropoff_longitude')
plt.savefig('bp of test_dropoff_longitude.png')
plt.show()


# In[34]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=test_cab['dropoff_latitude'],data=train_cab,orient='v')
plt.title('Boxplot of test_dropoff_latitude')
plt.savefig('bp of test_dropoff_latgitude.png')
plt.show()


# In[35]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['pickup_longitude'],data=train_cab,orient='v')
plt.title('Boxplot of pickup_longitude')
plt.savefig('bp of pickup_longitude.png')
plt.show()


# In[36]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['pickup_latitude'],data=train_cab,orient='v')
plt.title('Boxplot of pickup_latitude')
plt.savefig('bp of pickup_latitude.png')
plt.show()


# In[37]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['dropoff_longitude'],data=train_cab,orient='v')
plt.title('Boxplot of dropoff_longitude')
plt.savefig('bp of dropoff_longitude.png')
plt.show()


# In[38]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['dropoff_latitude'],data=train_cab,orient='h')
plt.title('Boxplot of dropoff_latitude')
#plt.savefig('bp of dropoff_latitude.png')
plt.show()


# In[39]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['passenger_count'],data=train_cab,orient='h')
plt.title('Boxplot of passenger_count')
#plt.savefig('bp of passenger_count.png')
plt.show()


# In[40]:


#  outliers in column 1:fare_amount
q75=np.percentile(train_cab.fare_amount,75)
q25=np.percentile(train_cab.fare_amount,25)
iqr=q75-q25
min=q25-(iqr*1.5)
#since fare_amount can not go negative so we will keep the minimum  val= 0
min=0
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)


# In[41]:


len(train_cab.loc[train_cab['fare_amount'] < min])


# In[42]:


len(train_cab.loc[train_cab['fare_amount']>max])


# In[43]:


train_cab.loc[train_cab['fare_amount'] < min,'fare_amount']


# In[44]:


train_cab.loc[train_cab['fare_amount'] < min,'fare_amount'] =np.nan
train_cab.loc[train_cab['fare_amount'] > max,'fare_amount'] =np.nan


# In[45]:


train_cab.loc[train_cab['fare_amount'] < min,'fare_amount']


# In[46]:


train_cab.fare_amount.isna().sum()
df=train_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
train_cab.fare_amount=df.fare_amount
train_cab.fare_amount.isna().sum()


# In[47]:


(1399/16067)*100


# In[48]:


train_cab.head(10)


# In[49]:


# outliers analysis for 3rd column
q75=np.percentile(train_cab.pickup_longitude,75)
q25=np.percentile(train_cab.pickup_longitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)


# In[50]:


len(train_cab.loc[train_cab['pickup_longitude'] < min])


# In[51]:


len(train_cab.loc[train_cab['pickup_longitude']>max])


# In[52]:


train_cab.loc[train_cab['pickup_longitude'] < min,'pickup_longitude']


# In[53]:


train_cab.loc[train_cab['pickup_longitude'] < min,'pickup_longitude']=np.nan
train_cab.loc[train_cab['pickup_longitude'] > max,'pickup_longitude']=np.nan


# In[54]:


train_cab.loc[train_cab['pickup_longitude'] < min,'pickup_longitude']


# In[55]:


train_cab.fare_amount.isna().sum()
df=train_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
train_cab.pickup_longitude=df.pickup_longitude
train_cab.pickup_longitude.isna().sum()


# In[56]:


#outlier analysis for test data  for pickup_longitude column
q75=np.percentile(test_cab.pickup_longitude,75)
q25=np.percentile(test_cab.pickup_longitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)

test_cab.loc[test_cab['pickup_longitude'] < min,'pickup_longitude']=np.nan
test_cab.loc[test_cab['pickup_longitude'] > max,'pickup_longitude']=np.nan

test_cab.pickup_longitude.isna().sum()
df=test_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
test_cab.pickup_longitude=df.pickup_longitude
test_cab.pickup_longitude.isna().sum()


# In[57]:


#outlier analysis for test data  for pickup_latitude column
q75=np.percentile(test_cab.pickup_latitude,75)
q25=np.percentile(test_cab.pickup_latitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)

test_cab.loc[test_cab['pickup_latitude'] < min,'pickup_latitude']=np.nan
test_cab.loc[test_cab['pickup_latitude'] > max,'pickup_latitude']=np.nan

test_cab.pickup_latitude.isna().sum()
df=test_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
test_cab.pickup_latitude=df.pickup_latitude
test_cab.pickup_latitude.isna().sum()


# In[58]:


#outlier analysis for test data  for dropoff_longitude column
q75=np.percentile(test_cab.dropoff_longitude,75)
q25=np.percentile(test_cab.dropoff_longitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)

test_cab.loc[test_cab['dropoff_longitude'] < min,'dropoff_longitude']=np.nan
test_cab.loc[test_cab['dropoff_longitude'] > max,'dropoff_longitude']=np.nan

test_cab.dropoff_longitude.isna().sum()
df=test_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
test_cab.dropoff_longitude=df.dropoff_longitude
test_cab.dropoff_longitude.isna().sum()


# In[59]:


#outlier analysis for test data  for dropoff_latitude column
q75=np.percentile(test_cab.dropoff_latitude,75)
q25=np.percentile(test_cab.dropoff_latitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)

test_cab.loc[test_cab['dropoff_latitude'] < min,'dropoff_latitude']=np.nan
test_cab.loc[test_cab['dropoff_latitude'] > max,'dropoff_latitude']=np.nan

test_cab.dropoff_latitude.isna().sum()
df=test_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
test_cab.dropoff_latitude=df.dropoff_latitude
test_cab.dropoff_latitude.isna().sum()


# In[60]:


# removing outliers
q75=np.percentile(train_cab.pickup_latitude,75)
q25=np.percentile(train_cab.pickup_latitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)


# In[61]:


len(train_cab.loc[train_cab['pickup_latitude'] < min])


# In[62]:


len(train_cab.loc[train_cab['pickup_latitude']>max])


# In[63]:


train_cab.loc[train_cab['pickup_latitude'] < min,'pickup_latitude']


# In[64]:


train_cab.loc[train_cab['pickup_latitude'] > max,'pickup_latitude']


# In[65]:


train_cab['pickup_latitude'].max()


# In[66]:


train_cab.loc[train_cab['pickup_latitude'] < min,'pickup_latitude']=np.nan
train_cab.loc[train_cab['pickup_latitude'] > max,'pickup_latitude']=np.nan
train_cab['pickup_latitude'].max()


# In[67]:


train_cab.loc[train_cab['pickup_latitude'] < min,'pickup_latitude']


# In[68]:


train_cab.fare_amount.isna().sum()
df=train_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
train_cab.pickup_latitude=df.pickup_latitude
train_cab.pickup_latitude.isna().sum()


# In[69]:


# removing outliers
q75=np.percentile(train_cab.dropoff_latitude,75)
q25=np.percentile(train_cab.dropoff_latitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)


# In[70]:


len(train_cab.loc[train_cab['dropoff_latitude'] < min])


# In[71]:


len(train_cab.loc[train_cab['dropoff_latitude'] > max])


# In[72]:


train_cab['dropoff_latitude'].max()


# In[73]:



train_cab.loc[train_cab['dropoff_latitude'] < min,'dropoff_latitude']=np.nan
train_cab.loc[train_cab['dropoff_latitude'] > max,'dropoff_latitude']=np.nan


# In[74]:


train_cab.fare_amount.isna().sum()
df=train_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
train_cab.dropoff_latitude=df.dropoff_latitude
train_cab.dropoff_latitude.isna().sum()


# In[75]:


# removing outliers
q75=np.percentile(train_cab.dropoff_longitude,75)
q25=np.percentile(train_cab.dropoff_longitude,25)
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
print(q75,q25,iqr,min,max)


# In[76]:


len(train_cab.loc[train_cab['dropoff_longitude'] < min])


# In[77]:


len(train_cab.loc[train_cab['dropoff_longitude'] > max])


# In[78]:


train_cab.loc[train_cab['dropoff_longitude'] < min,'dropoff_longitude']=np.nan
train_cab.loc[train_cab['dropoff_longitude'] > max,'dropoff_longitude']=np.nan


# In[79]:


train_cab.loc[train_cab['dropoff_longitude'] < min,'dropoff_longitude']


# In[80]:


train_cab.fare_amount.isna().sum()
df=train_cab.drop('pickup_datetime',axis=1)
df=pd.DataFrame(KNN(k=5).fit_transform(df),columns=df.columns)
train_cab.dropoff_longitude=df.dropoff_longitude
train_cab.dropoff_longitude.isna().sum()


# In[81]:


# column 6 is a passenger count which is a categorical variable so and has values from 1to 6
min=1
max=6
len(train_cab.loc[train_cab['passenger_count'] < min])
len(train_cab.loc[train_cab['passenger_count'] >= max])
train_cab.loc[train_cab['passenger_count'] < min,'passenger_count']=np.nan
train_cab.loc[train_cab['passenger_count'] > max,'passenger_count']=np.nan
train_cab.passenger_count.describe()
train_cab['passenger_count']=train_cab['passenger_count'].fillna(train_cab['passenger_count'].median())
sum(train_cab.passenger_count.isna())


# In[82]:


1132+42+241+766+86+705+1095+19+1399+3


# In[83]:


(5488/16067)*100


# In[84]:


train_cab.info()


# In[85]:



plt.figure(figsize=(20,10))
plt.xlim(0,100)
_ = sns.boxplot(x=train_cab['fare_amount'],y=train_cab['passenger_count'],data=train_cab,orient='h')
plt.title('Boxplot of fare_amount w.r.t passenger_count')
# plt.savefig('Boxplot of fare_amount w.r.t passenger_count.png')
plt.show()


# In[86]:


train_cab.describe()


# In[87]:


test_cab.describe()


# In[88]:


#data is outlier and missing value free
train_cab.to_csv('ntrain_cab.csv',index=False)
test_cab.to_csv('ntest_cab.csv',index=False)


# # feature engineering on latitude and longitude and on datetime variables

# In[89]:


# for train_cab 
#train_cab['years']=train_cab.pickup_datetime.dt.year
#train_cab['months']=train_cab.pickup_datetime.dt.month
#train_cab['weekday']=train_cab.pickup_datetime.dt.weekday


# In[90]:


# Calculate distance the cab travelled from pickup and dropoff location using great_circle from geopy library
data = [train_cab, test_cab]
for i in data:
    i['years']=i.pickup_datetime.dt.year
    i['months']=i.pickup_datetime.dt.month
    i['weekday']=i.pickup_datetime.dt.weekday
    i['great_circle']=i.apply(lambda x: great_circle((x['pickup_latitude'],x['pickup_longitude']),(x['dropoff_latitude'],x['dropoff_longitude'])).miles, axis=1)
    i['geodesic']=i.apply(lambda x: geodesic((x['pickup_latitude'],x['pickup_longitude']),(x['dropoff_latitude'],x['dropoff_longitude'])).miles, axis=1)


# In[91]:


train_cab.head()
test_cab.head()


# In[92]:


# for test_cab 
##test_cab['years']=test_cab.pickup_datetime.dt.year
#test_cab['months']=test_cab.pickup_datetime.dt.month
#test_cab['weekday']=test_cab.pickup_datetime.dt.weekday


# In[93]:


train_cab=train_cab.loc[:,['fare_amount', 'passenger_count', 'years',
       'months', 'weekday', 'geodesic']]
test_cab=test_cab.loc[:,['passenger_count', 'years',
       'months', 'weekday', 'geodesic']]


# In[94]:


print(train_cab.head())
print(test_cab.head())


# In[95]:


#data = [train_cab, test_cab]
#for i in data:
#    i['passenger_count']=i['passenger_count'].round().astype('object').astype('category',ordered=True)
#    i['years']=i['years'].round().astype('object').astype('category',ordered=False)
#    i['months']=i['months'].round().astype('object').astype('category',ordered=False)
#    i['weekday']=i['weekday'].round().astype('object').astype('category',ordered=False)


# In[96]:


train_cab.info()


# In[97]:


test_cab.info()


# In[98]:


#data is outlier and missing value free
train_cab.to_csv('nntrain_cab.csv',index=False)
test_cab.to_csv('nntest_cab.csv',index=False)


# # implementing  model on the data

# In[99]:


train_cab.head()


# In[100]:


#split data
# divide the data
train,test=train_test_split(train_cab,test_size=0.2)


# ## linear regression

# In[101]:


model=sm.OLS(train.iloc[:,0],train.iloc[:,1:7]).fit()


# In[102]:


model.summary()


# In[103]:


prediction_lr=model.predict(test.iloc[:,1:7])


# In[104]:


def MAPE(y_true,y_pred):
    mape = np.mean(np.abs((y_true-y_pred)/y_true))
    return mape


# In[105]:


MAPE(test.iloc[:,0],prediction_lr)


# ## decision tree

# In[106]:


#decision tree for regression
fit_dt=DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,1:7],train.iloc[:,0])


# In[107]:


#apply the model on test data
prediction_dt=fit_dt.predict(test.iloc[:,1:7])


# In[108]:


# calculate mape
def MAPE(y_true,y_pred):
    mape=np.mean(np.abs((y_true-y_pred)/y_true))
    return mape


# In[109]:


MAPE(test.iloc[:,0],prediction_dt)


# ## random forest

# In[110]:


# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 


# In[111]:fit the regressor with x and y data 


# fit the regressor with x and y data 
fit_rf=regressor.fit(train.iloc[:,1:7],train.iloc[:,0])   


# In[112]:


prediction_rf=fit_rf.predict(test.iloc[:,1:7])


# In[113]:


# calculate mape
def MAPE(y_true,y_pred):
    mape=np.mean(np.abs((y_true-y_pred)/y_true))
    return mape


# In[114]:


MAPE(test.iloc[:,0],prediction_rf)

