#  import packages
import numpy as np
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
# load the data
df = pd.read_csv(path)

# data shape
print('=================df shape========================')
print(df.shape)

print('=================df info========================')
print(df.info())

# remove special characters(',' & '$')from df columns
df['INCOME']=df['INCOME'].str.replace(',|\$', '')
df['HOME_VAL']=df['HOME_VAL'].str.replace(',|\$', '')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace(',|\$', '')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace(',|\$', '')
df['CLM_AMT']=df['CLM_AMT'].str.replace(',|\$', '')

# seperate features and target

#X= df.drop('CLAIM_FLAG',axis=1)
#y= df['CLAIM_FLAG']
#or we can use below
X= df.iloc[:, :-1]
y=df.iloc[:, -1]


# see shape of X & y
print('===== X & y shape =====')
print(X.shape,y.shape)


# target value count
count= y.value_counts()
print('target value count : ', count)


# split data in train & test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=6,test_size=0.30)

# Code ends here


# --------------
# Code starts here

# convert train & test set columns to flaot types
X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float) 


X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float) 

# check if null values available in training & testing sets

print(X_train.isnull().sum())
print('====Test Data Set====')
print(X_test.isnull().sum())

# Code ends here


# --------------

# drop missing values
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)


y_train=y_train[X_train.index]
y_test=y_test[X_test.index]



# fill missing values with mean
X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_test['AGE'].fillna((X_train['AGE'].mean()), inplace=True)

X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)



X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_test['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)



X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)


print(X_train.isnull().sum())
print(X_test.isnull().sum())


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here

# fit transform above columns using LabelEncoder
for col in columns:
  le = LabelEncoder()
  X_train[col]=le.fit_transform(X_train[col].astype(str))
  X_test[col]=le.fit_transform(X_test[col].astype(str))
  


# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

# create model object
model = LogisticRegression(random_state=6)

# fit the model
model.fit(X_train,y_train)

# predict using model
y_pred = model.predict(X_test)

# calculate the accuaracy
score = accuracy_score(y_test,y_pred)

# print accuracy
print('accuracy_score : ', score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

# create SMOTE objct
smote = SMOTE(random_state = 9)

# fit sample training data
X_train,y_train = smote.fit_sample(X_train,y_train)

# instantiate std scalr obj
scaler = StandardScaler()

# fit_transform training & transform test data using standard scalar
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here

# create model
model =LogisticRegression()

# train the model
model.fit(X_train,y_train)

# prediction using model
y_pred = model.predict(X_test)

# find the accuracy of model after applying SMOTE oversampling on imbalanced dataset
score = accuracy_score(y_test,y_pred)
print('accuracy score after SMOTE oversampling is :', score)
# Code ends here


