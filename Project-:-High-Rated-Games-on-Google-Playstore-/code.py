  # --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

#Loading the data
data=pd.read_csv(path)

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')

plt.show()


#Subsetting the dataframe based on `Rating` column
data=data[data['Rating']<=5]

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')   

#Code ends here


# --------------
#Code starts here

#Sum of null values of each column
total_null = data.isnull().sum()

#Percentage of null values of each column
percent_null = (total_null/data.isnull().count())

#Concatenating total_null and percent_null values
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])

print(missing_data)

#Dropping the null values
data.dropna(inplace = True)

#Sum of null values of each column
total_null_1 = data.isnull().sum()

#Percentage of null values of each column
percent_null_1 = (total_null_1/data.isnull().count())

#Concatenating total_null and percent_null values
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total', 'Percent'])

print(missing_data_1)

#Code ends here


# --------------

#Code starts here

sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)

plt.title('Rating vs Category [BoxPlot]')
plt.xticks(rotation=90)
plt.xticks()
plt.show()

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

print(data['Installs'].value_counts())

# replace special chars
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].str.replace(',','')

# convert to int
data['Installs']=data['Installs'].astype('int')

# Leabel Encoding
le = LabelEncoder()
data['Installs']= le.fit_transform(data['Installs'])

# plot the graph
sns.regplot(x="Installs", y="Rating" , data=data)
plt.title('Rating vs Installs [RegPlot]')
plt.show()
#Code ends here



# --------------
#Code starts here

#Code starts here

print(data['Price'].value_counts())

# replace special chars
data['Price']=data['Price'].str.replace('$','')

# convert to folat
data['Price']=data['Price'].astype('float')

# plot the graph
sns.regplot(x="Price", y="Rating" , data=data)
plt.title('Rating vs Price [RegPlot]')
plt.show()
#Code ends here




# --------------

#Code starts here

print(data['Genres'].unique())

print('-'*50)

data['Genres'] = data['Genres'].str.split(';').str[0]

gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

gr_mean = gr_mean.sort_values(by='Rating')

print('First Values of gr_mean : ', gr_mean.iloc[0])
print('Last Values of gr_mean : ', gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here

#Converting the column into datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#Creating new column having `Last Updated` in days
data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated'] ).dt.days 

#Setting the size of the figure
plt.figure(figsize = (10,10))

#Plotting a regression plot between `Rating` and `Last Updated Days`
sns.regplot(x="Last Updated Days", y="Rating", color = 'lightpink',data=data )

#Setting the title of the plot
plt.title('Rating vs Last Updated [RegPlot]',size = 20)

#Code ends here


