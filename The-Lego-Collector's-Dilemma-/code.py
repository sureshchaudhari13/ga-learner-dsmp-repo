# --------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# code starts here
df= pd.read_csv(path)
df.head()

X=df.drop('list_price',axis=1)
y = df.list_price

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=6,test_size=.30)


# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        

cols = X_train .columns

fig ,axes = plt.subplots(nrows = 3 , ncols = 3,figsize = (20, 15))



for i in range(0,3):
  for j in range(0,3):
    col=cols[ i * 3 + j]
    axes[i,j].scatter(X_train[col],y_train)
    axes[i,j].set_title(col)

plt.show()  



# code ends here



# --------------
# Code starts here

corr = X_train.corr()
corr

X_train = X_train.drop(columns=['play_star_rating','val_star_rating'],axis=1)
X_test = X_test.drop(columns=['play_star_rating','val_star_rating'],axis=1)

# Code ends here x


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

# Create the model
regressor=LinearRegression()

# model training
regressor.fit(X_train,y_train)

# model prediction
y_pred= regressor.predict(X_test)

# Calculate mse
mse = mean_squared_error(y_test,y_pred)
print('mse : ', mse)

# Calculate r2 score
r2 = r2_score(y_test,y_pred)
print('r2 : ', r2)

# Code ends here


# --------------
# Code starts here

residual = y_test - y_pred

plt.hist(residual)


# Code ends here


