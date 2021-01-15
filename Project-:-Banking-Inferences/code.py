# import packages 
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  

# Read Data
data = pd.read_csv(path)

# create Data sample
data_sample = data.sample(n=sample_size,random_state=0)
print('data_sample.shape :',data_sample.shape)

# Sample installment mean
sample_mean = data_sample['installment'].mean()
print('sample_mean :',sample_mean)

# Sample installment std deviation
sample_std = data_sample['installment'].std()
print('sample_std :',sample_std)

# Margin of Err
margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))
print('margin_of_error :',margin_of_error)

#Confidence  Interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print('confidence_interval :',confidence_interval)

# True Mean of population
true_mean = data['installment'].mean()
print( 'true_mean :',true_mean)






#Code starts here



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here


fig ,axes = plt.subplots(3,1, figsize=(15,8))

for i in range(len(sample_size)):
  m=[]  
  for j in range(1000):

        mean=data['installment'].sample(sample_size[i]).mean()
        m.append(mean)

  mean_series=pd.Series(m)
  mean_series.plot(kind='hist',ax=axes[i])


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
# Remove % from interest col values
data['int.rate'] = data['int.rate'].map(lambda x: str(x)[:-1])

data['int.rate']= data['int.rate'].astype(float)/100

# Apply Z-Test 
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(), alternative='larger')

print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

if p_value<0.05:
    inference ='Reject'
else:
    inference ='Accept'
  

print('inference :',inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here

# Apply Z-Test 
z_statistic, p_value = ztest(data[data['paid.back.loan']=='No']['installment'],data[data['paid.back.loan']=='Yes']['installment'])

print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

if p_value<0.05:
    inference ='Reject'
else:
    inference ='Accept'
  

print('inference :',inference)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here


# Subsetting the dataframe
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()


#Concating yes and no into a single dataframe
observed=pd.concat([yes.transpose(),no.transpose()], 1,keys=['Yes','No'])

print(observed)

chi2, p, dof, ex = chi2_contingency(observed)


print("Critical value")
print(critical_value)


print("Chi Statistic")
print(chi2)

#Code ends here


