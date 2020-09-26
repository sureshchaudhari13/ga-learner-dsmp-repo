# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Code starts here

# Read the data
data = pd.read_csv(path)

data.head()

# see loan status count
loan_status = data.Loan_Status.value_counts()
print('loan_status in  :', loan_status)

# Plot the Bar Chart for this.
loan_status.plot(kind='bar',grid=True,title='Bar Plot for Loan Approved Vs Not Approved',legend=True,figsize=(5,4))
plt.show()




# --------------
#Code starts here
#Code starts here

#Plotting an unstacked bar plot
property_and_loan=data.groupby(['Property_Area', 'Loan_Status'])
property_and_loan=property_and_loan.size().unstack()
property_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))

#Changing the x-axis label
plt.xlabel('Property_Area')

#Changing the y-axis label
plt.ylabel('Loan_Status')

#Rotating the ticks of X-axis
plt.xticks(rotation=45)

#Code ends here


# --------------
#Code starts here

education_and_loan=data.groupby(['Education', 'Loan_Status'])
education_and_loan=education_and_loan.size().unstack()
education_and_loan.plot(kind='bar', stacked=True, figsize=(15,10))

#Changing the x-axis label
plt.xlabel('Education Status')

#Changing the y-axis label
plt.ylabel('Loan Status')

#Rotating the ticks of X-axis
plt.xticks(rotation=45)

plt.show()


# --------------
#Code starts here
graduate = data[data['Education'] == 'Graduate']
not_graduate =data[data['Education'] == 'Not Graduate']

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,7))
graduate['LoanAmount'].plot(kind='density',title='Graduate',ax=ax1)

not_graduate['LoanAmount'].plot(kind='density',title='Not Graduate',ax=ax2)

plt.show()











#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here

fig ,(ax_1,ax_2,ax_3) = plt.subplots(nrows = 3 , ncols = 1,figsize=(10,7))

ax_1.scatter(data['ApplicantIncome'],data["LoanAmount"])
ax1.xlabel='Applicant Income'

ax_2.scatter(data['CoapplicantIncome'],data["LoanAmount"])
plt.xlabel='Coapplicant Income'

data['TotalIncome'] = data.ApplicantIncome + data.CoapplicantIncome
ax_3.scatter(data['TotalIncome'],data["LoanAmount"])
plt.xlabel='Total Income'

plt.show()




