# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path of the data file- path
data =pd.read_csv(path)

data.Gender.replace('-','Agender',inplace=True)

gender_count =data.Gender.value_counts()
print('gender_count :',gender_count)

# Barplot for Gender count
gender_count.plot(kind='bar',figsize=(10,8),title='Gender',grid=True,legend=True)

#Code starts here 




# --------------
#Code starts here

# vlaue count for Allignment
alignment  = data.Alignment.value_counts()
print('Alignment  :',alignment )

# Pie chart for Allignment
alignment.plot(kind='pie',figsize=(10,8),title='Character Alignment',grid=True,legend=True)


# --------------
#Code starts here

sc_df = data[['Strength','Combat']]

# Covariance of Strength','Combat'
sc_covariance= sc_df['Strength'].cov(sc_df['Combat'])
print('sc_covariance :',sc_covariance)

# std deviation of Strength','Combat'
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()

# Pearson corr coeff of Strength','Combat'
sc_pearson = sc_covariance / (sc_strength * sc_combat)
print('Pearson Corr Coeff :', sc_pearson)

print('---------------------------------------------------------')

ic_df = data[['Intelligence','Combat']]

# Covariance of Intelligence','Combat'
ic_covariance= ic_df['Intelligence'].cov(ic_df['Combat'])
print('ic_covariance :',ic_covariance)

# std deviation of Intelligence','Combat'
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()

# Pearson corr coeff of Intelligence','Combat'
ic_pearson = ic_covariance / (ic_intelligence * ic_combat)
print('Pearson Corr Coeff :', ic_pearson)


# --------------
#Code starts here

# quantile=0.99 for the column Total
total_high = data['Total'].quantile(q=0.99)
print('total_high (90% quantile): ',total_high)

# data subject for Total > total_high
super_best = data[data['Total']>total_high]

# top superheroes/villains
super_best_names = list(super_best['Name'])
print('super_best_names (Super Heros are :)',super_best_names)


# --------------
#Code starts here

fig,(ax_1,ax_2,ax_3) = plt.subplots(1,3,figsize=(20,8))

super_best['Intelligence'].plot(kind='box',ax=ax_1)
ax_1.set_title('Intelligence')

super_best['Speed'].plot(kind='box',ax=ax_2)
ax_1.set_title('Speed')

super_best['Power'].plot(kind='box',ax=ax_3)
ax_1.set_title('Power')
plt.show()


