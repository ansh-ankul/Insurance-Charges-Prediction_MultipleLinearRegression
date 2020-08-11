import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset= pd.read_csv("E:/Machine Learning (ML)/Projects/Health insurance Simple Linear Regression/insurance.csv")

dataset.shape

dataset.info()

dataset.describe()


#--------------------------------------Data Visualization----------------------------------------------------------
plt.figure(figsize=(20,8))

#----------------Charges Distribution plot
plt.subplot(1,2,1)
plt.title('Insurance Charges Distribution Plot')
sns.distplot(dataset.charges)
# Flexibly plot a univariate distribution of observations.


plt.subplot(1,2,2)
plt.title('Insurance charges Spread')
sns.boxplot(y=dataset.charges)
# In descriptive statistics, a box plot or boxplot
# is a method for graphically depicting groups of numerical data through their quartiles.
plt.show()

#---------------Sex plot
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = dataset.sex.value_counts().plot(kind='bar')
plt.title('Sex Histogram')
plt1.set(xlabel = 'Sex', ylabel='Frequency')

plt.show()

df = pd.DataFrame(dataset.groupby(['sex'])['charges'].mean().sort_values(ascending = False))
df.plot.bar(color='orange')
plt.title('sex vs Average Charges')
plt.show()

#---------------Region plot
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = dataset.region.value_counts().plot(kind='bar')
plt.title('Region Histogram')
plt1.set(xlabel = 'Region', ylabel='Frequency')

plt.show()


#---------------Bmi vs Charges
dataset["bmi_range"] = dataset['bmi'].apply(lambda x : "thin" if x < 19
                                                     else ("fit" if  19 <= x < 25
                                                           else ("overweight" if  25 <= x < 28
                                                                else ("Obese"))))
df = pd.DataFrame(dataset.groupby(['bmi_range'])['charges'].mean().sort_values(ascending = False))
df.plot.bar(color='orange')
plt.title('bmi-range vs Average Charges')
plt.show()


#---------------Age plot
#setting up levels for price.
dataset["age_range"] = dataset['age'].apply(lambda x : "low" if x < 25 
                                                     else ("Medium" if 25 <= x < 40
                                                           else ("High")))
dataset.head()
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = dataset.age_range.value_counts().plot(kind='bar')
plt.title('age_range Histogram')
plt1.set(xlabel = 'age_range', ylabel='Frequency')

plt.show()


#---------------No of children
fig, ax = plt.subplots(figsize = (15,5))
plt1 = sns.countplot(dataset['children'], order=pd.value_counts(dataset['children']).index,)
plt1.set(xlabel = 'No of children', ylabel= 'No of children')
plt.show()
plt.tight_layout()


#--------------Smoker or not
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Smoker Histogram')
sns.countplot(dataset.smoker, palette=("RdBu"))

plt.subplot(1,2,2)
plt.title('Smoking vs Price')
sns.boxplot(x=dataset.smoker, y=dataset.charges, palette=("RdBu"))

plt.show()

df = pd.DataFrame(dataset.groupby(['smoker'])['charges'].mean().sort_values(ascending = False))
df.plot.bar(color='orange')
plt.title('smoker vs Average Charges')
plt.show()

#Significant features are: sex, bmi_range, age_range, number, smoker
#Dropping nonimportant features
dataset = dataset.drop(["region"], axis = 1)
dataset = dataset.drop(["age"], axis = 1)
dataset = dataset.drop(["bmi"], axis = 1)


#Converting numbers into string in column 'children'
def replace_name(a,b):
    dataset.children.replace(a,b,inplace=True)

replace_name(0,'0children')
replace_name(1,'1children')
replace_name(2,'2children')
replace_name(3,'3children')
replace_name(5,'5children')
replace_name(6,'6children')

#------------------------------------------Model Building------------------------------------------------------

attributes = dataset[['age_range','sex','bmi_range','children','smoker','charges']]
attributes.head()

y=dataset.iloc[:,3].values


#Handling Categorical Data
# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

attributes = dummies('age_range',attributes)
attributes = dummies('sex',attributes)
attributes = dummies('bmi_range',attributes)
attributes = dummies('smoker',attributes)
attributes = dummies('children',attributes)

#Splitting into training and test set
from sklearn.model_selection import train_test_split
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(attributes,y, train_size = 0.8, test_size = 0.2, random_state = 100)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars=['charges']
x_train[num_vars] = scaler.fit_transform(x_train[num_vars])

y_train = x_train.pop('charges')

import statsmodels.api as sm
model = sm.OLS(y_train, x_train.astype(float)).fit()
model.summary()

def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X

X_train_new = build_model(x_train.astype(float),y_train)
X_train_new = x_train.drop(['male','0children','1children','2children','3children','5children','thin'], axis = 1)


X_train_new = build_model(X_train_new.astype(float),y_train)

lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 


num_vars = ['charges']
x_test[num_vars] = scaler.fit_transform(x_test[num_vars])

#Dividing into X and y
y_test = x_test.pop('charges')


# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
x_test_new = x_test[X_train_new.columns]

# Adding a constant variable 
x_test_new = sm.add_constant(x_test_new)


y_pred = lm.predict(x_test_new.astype(float))


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)  

