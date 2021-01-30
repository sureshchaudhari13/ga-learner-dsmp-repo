    # import libraries
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
# Read the data
df = pd.read_csv(filepath_or_buffer=path,compression='zip')
print(df.head())

# Dependent variable
y = df['attr1089']

# Independent variable
X = df.drop(columns=['attr1089'])

# Split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)

# Standardize the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_test[45,5])

# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# instantiate LogisticRegression model
lr = LogisticRegression()

# fit & predict the values using model
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

# find the roc_score
roc_score = roc_auc_score(y_test,y_pred)
print('roc_score : ',roc_score)


# ------------
from sklearn.tree import DecisionTreeClassifier

# create model
dt = DecisionTreeClassifier(random_state=4)

# fit & predict using model
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

# find the roc_auc_score for DTree
roc_score = roc_auc_score(y_test,y_pred)
print('roc_score using DTree : ',roc_score)


# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here

# Create RF model
rfc = RandomForestClassifier(random_state=4)

# fit & predict using model
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

# find the roc_auc_score score
roc_score = roc_auc_score(y_test,y_pred)
print('roc_score using DTree : ',roc_score)



# Code ends here

# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here
bagging_clf = BaggingClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=100 , max_samples=100 , random_state=0)

# fit & predict using model
bagging_clf.fit(X_train,y_train)

# find the score
score_bagging = bagging_clf.score(X_test,y_test)
print('score_bagging : ',score_bagging)


# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
# create Voting classifier with hard voting
voting_clf_hard = VotingClassifier(estimators=model_list,voting='hard')



# fit & predict using model
voting_clf_hard.fit(X_train,y_train)

# find the score on test data using hard voting
hard_voting_score = voting_clf_hard.score(X_test,y_test)
print('hard_voting_score : ',hard_voting_score)

# Code ends here


