# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
# Read data from file
df = pd.read_csv(path)

# print first 5 rows
print(df.head())

# split datframe into features and targets
X= df.drop('insuranceclaim',axis=1)
y=df['insuranceclaim']

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=6)


# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here

#  boxplot for bmi
sns.boxplot(X_train['bmi'])

# get qunatile value for .95
q_value= X_train['bmi'].quantile(q=0.95)

# y_train value count to check the balanceness of the class
y_train.value_counts()

# Code ends here


# --------------
# Code starts here

# get features correlation
relation = X_train.corr()

print(relation)

# plot the pair plot to see all featues correlation
sns.pairplot(X_train)

# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols =['children','sex','region','smoker']
fig ,axes = plt.subplots(nrows = 2 , ncols = 2)

for i in range(2):
  for j in range(2):
    col= cols[i*2 + j]
    sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

plt.show()


# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
# create model
lr = LogisticRegression(random_state=9)

# create a grid 
grid = GridSearchCV(estimator=lr,param_grid=parameters)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_test)

# calculate the accuracy score
accuracy= accuracy_score(y_test,y_pred)
print('accuracy_score',accuracy)

# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here

# calculate roc_auc_score
score = roc_auc_score(y_test,y_pred)

# get the y_pred_proba
y_pred_proba = grid.predict_proba(X_test)[:,1]

# find tpr, fpr
fpr, tpr, _= metrics.roc_curve(y_test,y_pred)

# find roc_auc score
roc_auc = roc_auc_score(y_test,y_pred_proba)

# plot roc_auc curve
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.show()
# Code ends here


