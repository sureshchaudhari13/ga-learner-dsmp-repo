# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
#Code starts here

df= pd.read_csv(path)
print(df.head())
print ('-------------------------------------------------------------------')

# seperate Features & Target var
X = df.drop('Price',axis=1)
y=df['Price']

# Split the data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6)

# Find corr between features
corr = X_train.corr()
print('Corr: ',corr)



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here

# instantiate model
regressor = LinearRegression()

#  train model
regressor.fit(X_train,y_train)

# predictions usinf model
y_pred = regressor.predict(X_test)

# R2 score calculations
r2 = r2_score(y_test,y_pred)
print('r2 score :' ,r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here

# instantiate model
lasso = Lasso()

#  train model
lasso.fit(X_train,y_train)

# predictions usinf model
lasso_pred = lasso.predict(X_test)

# R2 score calculations
r2_lasso = r2_score(y_test,lasso_pred)
print('r2_lasso score :' ,r2_lasso)




# --------------
from sklearn.linear_model import Ridge

# Code starts here
# instantiate model
ridge = Ridge()

#  train model
ridge.fit(X_train,y_train)

# predictions usinf model
ridge_pred = ridge.predict(X_test)

# R2 score calculations
r2_ridge = r2_score(y_test,lasso_pred)
print('r2_ridge score :' ,r2_ridge)


# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here

# instantiate model
regressor = LinearRegression()

#  train model
score = cross_val_score(regressor, X_train, y_train, cv=10)

mean_score = score.mean()
print('mean_score :' ,mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

# create model with polynomial features
model = make_pipeline(PolynomialFeatures(2),LinearRegression())

#fit & Predict using model 
model.fit(X_train,y_train)
 
y_pred = model.predict(X_test)

r2_poly = r2_score(y_test,y_pred)
print('r2_poly :',r2_poly)



