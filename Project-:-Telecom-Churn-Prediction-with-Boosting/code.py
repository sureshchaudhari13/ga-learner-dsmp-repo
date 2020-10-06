# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here

# read data
df = pd.read_csv(path)

# seperate featues and target vars
X = df.drop(['customerID','Churn'],axis=1)
y=df['Churn']

# split the data into training and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here

#Replacing spaces with 'NaN' in train dataset
X_train['TotalCharges'].replace(' ',np.NaN, inplace=True)

#Replacing spaces with 'NaN' in test dataset
X_test['TotalCharges'].replace(' ',np.NaN, inplace=True)

#Converting the type of column from X_train to float
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)

#Converting the type of column from X_test to float
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

#Filling missing values
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(),inplace=True)
X_test['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace=True)

#Check value counts
print(X_train.isnull().sum())

cat_cols = X_train.select_dtypes(include='O').columns.tolist()

#Label encoding train data
for x in cat_cols:
    le = LabelEncoder()
    X_train[x] = le.fit_transform(X_train[x])

#Label encoding test data    
for x in cat_cols:
    le = LabelEncoder()    
    X_test[x] = le.fit_transform(X_test[x])

#Encoding train data target    
y_train = y_train.replace({'No':0, 'Yes':1})

#Encoding test data target
y_test = y_test.replace({'No':0, 'Yes':1})



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

# create AdaBoostClassifier model
ada_model = AdaBoostClassifier(random_state=0)

# fit the model
ada_model.fit(X_train,y_train)

# prediction using model
y_pred = ada_model.predict(X_test)


# print the evaluation matrix results
ada_score = accuracy_score(y_test,y_pred)
ada_cm = confusion_matrix(y_test,y_pred)
ada_cr = classification_report(y_test,y_pred)

print('accuracy_score :', ada_score)
print('confusion_matrix :', ada_cm)
print('classification_report :', ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}


# Code starts here

# create XGB model
xgb_model = XGBClassifier(random_state=0)

# fit XGB model
xgb_model.fit(X_train,y_train)

# predict values
y_pred = xgb_model.predict(X_test)

# print the evaluation matrix results using XGB model
xgb_score = accuracy_score(y_test,y_pred)
xgb_cm = confusion_matrix(y_test,y_pred)
xgb_cr = classification_report(y_test,y_pred)

print('accuracy_score XGB Model:', xgb_score)
print('confusion_matrix XGB Model:', xgb_cm)
print('classification_report XGB Model:', xgb_cr)


# create a GridSearch 
clf_model = GridSearchCV(estimator=xgb_model ,param_grid=parameters)

# fit GridSearch model
clf_model.fit(X_train,y_train)

y_pred = clf_model.predict(X_test)

# print the evaluation matrix results with GridSearch 
clf_score = accuracy_score(y_test,y_pred)
clf_cm = confusion_matrix(y_test,y_pred)
clf_cr = classification_report(y_test,y_pred)

print('accuracy_score XGB Model with GridSearch:', clf_score)
print('confusion_matrix XGB Model with GridSearch:', clf_cm)
print('classification_report XGB Model with GridSearch :', clf_cr)












