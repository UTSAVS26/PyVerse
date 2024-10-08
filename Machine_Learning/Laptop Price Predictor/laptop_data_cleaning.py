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














