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














