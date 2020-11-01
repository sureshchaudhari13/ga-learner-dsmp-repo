# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df= pd.read_csv(path)

p_a = len(df[df['fico']>700])/(df.shape[0])

p_b = len(df[df['purpose'] == 'debt_consolidation'])/(df.shape[0])

df1= df[df['purpose'] == 'debt_consolidation']

P_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]

p_a_b= (p_b * P_a_b)/p_a

if P_a_b == p_a:
  result= True
else:
  result= False

print('result :',result)

# code ends here


# --------------
# code starts here

#Calculate the probability p(A)
prob_lp = len(df[df['paid.back.loan'] == 'Yes'])/(df.shape[0])
print('prob_lp:',prob_lp)


#Calculate the probability p(B)
prob_cs = len(df[df['credit.policy'] == 'Yes'])/(df.shape[0])
print('prob_cs :',prob_cs)

new_df = df[df['paid.back.loan'] == 'Yes']

prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]
print('prob_pd_cs : ',prob_pd_cs)

bayes = (prob_pd_cs * prob_lp)/ prob_cs

print('bayes : ',bayes)

# code ends here


# --------------
# code starts here

# Visualize the bar plot for the feature purpose
df['purpose'].value_counts().plot.bar()

# Create new df
df1 = df[df['paid.back.loan'] == 'No']

# Visualize the bar plot for the feature purpose on new df
df1['purpose'].value_counts().plot.bar()

# code ends here



# --------------
# code starts here
import matplotlib.pyplot as plt

inst_median= df['installment'].median()
print ('inst_median :',inst_median)

inst_mean= df['installment'].mean()
print('inst_mean : ',inst_mean)

# initialize the fig
fig, (ax_1, ax_2) = plt.subplots(2,1, figsize=(20,10))

df['installment'].plot(kind='hist',ax=ax_1)
ax_1.set_title('histogram for installment')

df['log.annual.inc'].plot(kind='hist',ax=ax_2)
ax_2.set_title('histogram for log.annual.inc')


# code ends here


