# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here

data = pd.read_csv(path)

print('*******data.head()*******')
print(data.head())

#split data into features and target
X= data.drop(['customer.id','paid.back.loan'],axis=1)
y=data['paid.back.loan']

#split data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
# distribution for 'paid.back.loan'
fully_paid  = y_train.value_counts()

# barplot for target 
fully_paid.plot(kind='bar',figsize=(8,5),legend=True,grid=True)
plt.title('Barplot for Target variable')
plt.xticks(rotation=45)
plt.show()

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
# modify 'int.rate' col values and convert to folat X_train
X_train['int.rate'] = X_train['int.rate'].str.replace('%','').astype('float')

# divide values by 100
X_train['int.rate'] = X_train['int.rate']/100

# modify 'int.rate' col values and convert to folat X_test
X_test['int.rate'] = X_test['int.rate'].str.replace('%','').astype('float')

X_test['int.rate'] = X_test['int.rate']/100

# select only numeric cols from training set
num_df= X_train.select_dtypes(include='number')

# select only categorical cols from training set
cat_df= X_train.select_dtypes(include='object')


# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here

cols = list(num_df.columns)

# plot boxplots for all numeric cols against target var
fig,axes = plt.subplots(nrows = 9 , ncols = 1,figsize=(10,20))
for i in range(len(cols)):
  sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])
  
plt.show()

# Code ends here


# --------------
# Code starts here

cols = list(cat_df.columns)

# plot countplot for categorical features
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(20,15))
for i in range(len(cols)-2):
  for j in range(len(cols)-2):
    sns.countplot(x=X_train[cols[i*2+j]], hue=y_train , ax=axes[i,j])
    sns.set_rotation=45
  
plt.show()  
  

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

#Code starts here

#Looping through categorical columns
for col in cat_df.columns:
    
    #Filling null values with 'NA'
    X_train[col].fillna('NA',inplace=True)
    
    #Initalising a label encoder object
    le=LabelEncoder()
    
    #Fitting and transforming the column in X_train with 'le'
    X_train[col]=le.fit_transform(X_train[col]) 
    
    #Filling null values with 'NA'
    X_test[col].fillna('NA',inplace=True)
    
    #Fitting the column in X_test with 'le'
    X_test[col]=le.transform(X_test[col]) 

# Replacing the values of y_train
y_train.replace({'No':0,'Yes':1},inplace=True)

# Replacing the values of y_test
y_test.replace({'No':0,'Yes':1},inplace=True)

#Initialising 'Decision Tree' model    
model=DecisionTreeClassifier(random_state=0)

#Training the 'Decision Tree' model
model.fit(X_train, y_train)

#Finding the accuracy of 'Decision Tree' model
acc=model.score(X_test, y_test)

#Printing the accuracy
print(acc)

#Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here

# create model
model_2 = DecisionTreeClassifier(random_state=0)

# create GridSearchCV
p_tree = GridSearchCV(estimator=model_2, param_grid=parameter_grid , cv=5)

# fit the model using GridSearchCV obj
p_tree.fit(X_train,y_train)

#caclulate the accuarcy
acc_2 =p_tree.score(X_test,y_test)
print('acc_2: ',acc_2)

# find the best model parameters
print ('p_tree.best_params_ :',p_tree.best_params_)


# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)


# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


