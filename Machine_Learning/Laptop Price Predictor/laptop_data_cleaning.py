import numpy as np
import pandas as pd

df = pd.read_csv("laptop_data.csv")
print(df.head())

print(df.shape)             # to check the shape of the dataframe
print(df.info( ))              # to check null values and datatype of a columns

print(df.duplicated().sum())         # to check whether two rows are not duplicated

print(df.isnull().sum())

df.drop(columns=["Unnamed: 0"],inplace=True)
print(df.head())

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

print(df.head(5))
print(df.info())

import seaborn as sns
import matplotlib.pyplot as plt

# df['Company'].value_counts().plot(kind='bar')

# sns.barplot(x= df['Company'] , y = df['Price'])
# plt.xticks(rotation = 'vertical')

# df['TypeName'].value_counts().plot(kind='bar')
#
# sns.barplot(x= df['TypeName'] , y = df['Price'])
# plt.xticks(rotation = 'vertical')
#
# plt.show()

# ---------------------------- Screen Resolution Adjuster -------------------

print("\n",df['ScreenResolution'].value_counts)

df['Touchscreen'] = df['ScreenResolution'].apply(lambda x :1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x :1 if 'IPS' in x else 0)

print(df.sample(5))


new = df['ScreenResolution'].str.split('x',n=1,expand = True)
df['new2'] = new[0]

df['width'] = df['new2'].str.split().str[-1]
df['height'] = new[1]

df['width'] = df['width'].astype('int32')
df['height'] = df['height'].astype('int32')

df.drop(columns=['new2'],inplace=True)
print(df.head())
print(df.info())

print(df.corr(numeric_only=True)['Price'])

print()

df['ppi'] =(df['width']**2 + df['height']**2)**0.5 / df['Inches']
print(df.head())
df.drop(columns = ['ScreenResolution','width','height'], inplace=True)
print(df.corr(numeric_only=True)['Price'])
print(df.head())

# ----------------------------------- CPU Name ----------------------------------

df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
print(df.head())
print(df['Cpu Name'].value_counts())

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3' or text == 'Intel Celeron Dual':
        return text
    else:
        if text.split()[0] == 'Intel' :
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
# print(df.sample(20))
df.drop(columns=['Cpu','Cpu Name'],inplace=True)
print(df.sample(5))


print(df['Memory'].sample(10))

# ======================================== SSD ========================

print(df['Memory'].head(24))

# print(df['Memory'].str.split('SSD').head(24))

df['test_ssd'] = df['Memory'].str.split('SSD')

df['test_ssd'] = df['test_ssd'].apply(lambda x:  x[0] if len(x) == 2 else 0)


df['test_ssd'] = df['test_ssd'].apply(lambda x:  x.split('GB')[0] if 'GB' in str(x) else x.split('TB')[0] if 'TB' in str(x) else 0)

df['SSD'] = df['test_ssd'].astype('float32')
df['SSD'] = df['SSD'].astype('int32')



# =============================================== hdd ========================
# print(df['Memory'].head(24))


df['test_hdd'] = df['Memory'].str.split('+' or 'HDD')

# print(df['test_hdd'].head(24))

df['test_hdd'] = df['test_hdd'].apply(lambda x:  x[-1] if len(x) == 2 else  x[0]  if "HDD" in x[0] else 0)
# print(df['test_hdd'].head(24))
# print(df['test_hdd'].apply(lambda x:  x.split('GB')[0] if 'GB' in str(x) else x.split('TB')[0] if 'TB' in str(x) else 0 ).head(24))

df['test_hdd'] = df['test_hdd'].apply(lambda x:  x.split('GB')[0] if 'GB' in str(x) else x.split('TB')[0] if 'TB' in str(x) else 0 )
#
# print(df['test_hdd'].sample(24))
df['HDD'] = df['test_hdd'].astype('float32')
df['HDD'] = df['HDD'].astype('int32')
df['HDD'] = df['HDD'].apply(lambda x: x*1024 if x<10 else x)
# print(df['HDD'].head(24))


df.drop(columns=['test_ssd','test_hdd','Memory'],inplace=True)
# print(df['Memory'].head(24))
print(df.head(24))

print(df.corr(numeric_only=True)['Price'])


df['GPU brand'] = df['Gpu'].apply(lambda x:x.split()[0])
print(df['GPU brand'].value_counts())

df = df[df['GPU brand'] != 'ARM']

df.drop(columns=['Gpu'],inplace=True)

print(df.head())

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp=='Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp =='Mac OS X':
        return 'Mac'
    else:
        return 'Others/No Os/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'],inplace=True)
print(df.corr(numeric_only=True)['Price'])
print(df.columns)
print(df.head())


# sns.displot(df['Price'])      # skewed distribution the price is not evenly distributed in the table which can lead to false values
# plt.show()

# If the distribution is skewed, it means that the graph is not symmetrical. It might have a long tail on one side, indicating that there are a few prices that are much higher than the rest. This can happen if there are some expensive items in the dataset that are significantly more costly than the majority of items.
#
# Why is this a problem? Skewed distributions can impact data analysis and modeling in a few ways:
#
# Misleading Interpretations: Skewed data can give a distorted view of the overall pattern or average price. For example, the mean (average) price can be heavily influenced by a few extremely high-priced items, making it appear higher than what most items actually cost.

# sns.displot(np.log(df['Price']))
# plt.show()




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error

#import algorigthms now

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

df.drop(columns=['Inches'],inplace=True)

x = df.drop(columns=['Price'])
y = np.log(df['Price'])
# print(x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 2)

# print(df.columns)
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False, drop='first'),[0,1,7,10,11])   # enter indexes on which one  hot encoding to be applying
],remainder='passthrough')

# step2 = LinearRegression()
# step2 = Ridge(alpha=10)
# step2 = Lasso(alpha = 0.001)
# step2 = KNeighborsRegressor(n_neighbors=3)
# step2 = DecisionTreeRegressor(max_depth=4)
# step2 = SVR(kernel='rbf',C=10000, epsilon=0.1)
step2 = RandomForestRegressor(
    n_estimators=100,
    random_state=5,
    max_samples=0.7,
    max_features=0.15,
    max_depth=25
)
# step2 = ExtraTreesRegressor(
#     n_estimators=100,
#     random_state=5,
#     max_samples=0.5,
#     max_features=0.13,
#     max_depth=15,
#     bootstrap=True
# )
# step2 = AdaBoostRegressor(n_estimators=27,learning_rate=1.0)
# step2 = GradientBoostingRegressor(n_estimators=500)
# step2 = XGBRegressor(n_estimators = 40, max_depth=5 , learning_rate=0.6)


pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])


# -------------------- Linear Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))        # R2 score 0.8152837383011426

# -------------------- Ridge Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))   # R2 score 0.8199736728206772


# -------------------- Lasso Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))           # R2 score 0.8174040400155699

# -------------------- KNN Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))              # R2 score 0.7881961438778196

# -------------------- Decision Tree Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))     # R2 score 0.7797903009324011

# -------------------- SVM Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))        # R2 score 0.7403779423062873

# -------------------- Random forest Regression ------------------------

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))           # R2 score 0.8663323136335733  ---> R2 score 0.8881535660014555

# -------------------- Extra Tree Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))            # R2 score 0.8766441310883409

# -------------------- AdaBoost Regression ------------------------
#
# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))            # R2 score 0.7890329023897401

# -------------------- Gradient Boost Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))           # R2 score 0.8680754945662121

# -------------------- XGB Regression ------------------------

# pipe.fit(x_train,y_train)
# y_pred = pipe.predict(x_test)
#
# print('R2 score',r2_score(y_test,y_pred))           # R2 score 0.8521172187173272


import pickle

# pickle.dump(df,open('laptop_price_data.pkl','wb'))
df.to_csv("cleaned_laptop_price_data.csv")
pickle.dump(pipe,open('RandomForestModel.pkl','wb'))












