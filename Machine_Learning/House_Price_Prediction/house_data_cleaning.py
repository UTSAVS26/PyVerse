import joblib as joblib
import pandas as pd
import numpy as np

data = pd.read_csv("housedata.csv")

# print(data.head())
# print(data.columns)
# print(data.shape)
# print(data.info())


# view your data # data analysis        # to fix  1. null values   2.size ( bedroom and bhk )   3.total_sqft ( some values are in range )
# ================================================= 1. Null Values Fix =================================================

for columns in data.columns:
    print(data[columns].value_counts())
    print("*"*20)

# finding null values
# print(data.isna().sum()) # we found that the society and balcony is having more null values and is off less use for us so just drop bruh :)

data.drop(columns=["society", "balcony"], inplace=True)
# print(data.describe()) # it shows the data with float values and shows data mean


# print("Data",data.info())   # check data which s null values # we found that location, size and bath is having missing values

# check data for location where is maximum number of locations available and replace the empty data with that location
# print(data["location"].value_counts())          # checking the data for max locations
data["location"] = data["location"].fillna("Whitefield")    # we found whitefield to be max-data, so we replaced it with dat

# print(data["size"].value_counts())             # checking data for max size
data["size"] = data["size"].fillna("2 BHK")    # we found 2 to be max-data, so we replaced it with dat

data["bath"] = data["bath"].fillna(data["bath"].median())     # fill the bathroom empty data with its median values

# print(data.info())   # no null values found :)

# ====================================== 1. Null Values Fixed ========================================================


# -------------------------------------- 2. multi values in a single columns -----------------------------------------
# in size columns there are values like 2 Bedroom and 2 BHK so need to solve that

data["bhk"]  = data["size"].str.split().str.get(0).astype(int)
# print(data["size"])
# print(data["bhk"])

# print(data[data.bhk>20])      # checking is there any bhk is greater than 20 , ❌ this is outliers in data


# *************************************** fixing sqft range problem by replacing range with mean ********************

# print(data["total_sqft"].unique())

# convert range function
def convertRange(x):
    lst = x.split("-")
    if len(lst) == 2:
        return (float(lst[0]) + float(lst[1]))/2
    try:
        return float(x)
    except:
        return None
data["total_sqft"] = data["total_sqft"].apply(convertRange)   # apply will apply the function convert Range on all total sqft  by passing every single values to it

# print(data.columns)


# - - - -  - -- - - -  - dropping the non useful columns - - - - - - - - -
cols = ['area_type', 'availability','size',]
data.drop(columns=cols , inplace=True)
# print(data.head())
# --------------------------------------------------------------------------

# ------------------------------------- creating a new column of price per square feet -------------------------------

data["price_per_sqft"] = data["price"] * 100000  / data["total_sqft"]
# print(data.head())

# print(data.describe())

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Fixing Outliers $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# locations and bhk are having outliers

# print(data["location"].value_counts())        # there are total of 1305 locations which are not that easy to put in selection tab
# so we replace the location which are having count less than or equal to 10 with Others

data["location"] = data["location"].apply(lambda x: x.strip())   # removing the imperfect spaces that with this function
location_count = data["location"].value_counts()
# print("*"*20,location_count)   # location count is reduced because difference in white space was creating different locations

loc_less_than_10  = location_count[location_count<=10]  # check which locations is less than or equal to 10
# print(loc_less_than_10)

# replace those locations with the word Others

data["location"] = data["location"].apply(lambda x:"Other" if x in loc_less_than_10 else x)
# print(data["location"].value_counts()) # so we reduced the length of location to 242 values now from 1305



# print(data.describe())
# print((data["total_sqft"]/data["bhk"] ).describe())    # checking the data according to bhk wise

data = data[(data["total_sqft"]/data["bhk"] ) >=300]   # we keep the data of only those whose total_sqft/bhk is greater than 300
# print(data.describe())
# print(data.shape)
#
# print(data["price_per_sqft"].describe())         # max value outlier found

# solving max value outlier

def remove_outlier_sqft(df):
    final_df = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        mean = np.mean(subdf["price_per_sqft"])
        standard_deviation = np.std(subdf["price_per_sqft"])
        general_dataframe = subdf[(subdf["price_per_sqft"] > (mean - standard_deviation)) & (subdf["price_per_sqft"] <= (mean + standard_deviation))]
        final_df = pd.concat([final_df, general_dataframe], ignore_index=True)
    return final_df

data = remove_outlier_sqft(data)

# now removing outliers from bhk

def bhk_outlier_remover(df):
    exclue_indices = np.array([])
    for location , location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df["price_per_sqft"]),
                'std': np.std(bhk_df["price_per_sqft"]),
                'count': bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclue_indices = np.append(exclue_indices, bhk_df[bhk_df['price_per_sqft']<(stats['mean'])].index.values)
        return df.drop(exclue_indices, axis = 'index')

data = bhk_outlier_remover(data)

# print(data.describe())
# print(data.shape)
# print(data)

data.drop(columns =['price_per_sqft'], inplace = True)

# print(data)


# ________________________________________________________ Cleaned Data ______________________________________________
# -------------------------------------------------------- Saving Clean Data -----------------------------------------
data.to_csv("cleaned_housedata.csv")                                     # ⚠️⚠️⚠️ need to run only once

x = data.drop(columns=["price"])
y = data["price"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)


column_trans = make_column_transformer((OneHotEncoder(sparse_output=False) , ['location']),remainder='passthrough')
scaler = StandardScaler()



# LINEAR REGRESSION
linear_regression = LinearRegression()
pipe = make_pipeline(column_trans,scaler,linear_regression)

pipe.fit(x_train,y_train)

y_pred_lr  = pipe.predict(x_test)
print(r2_score(y_test, y_pred_lr))

# LASSO
lasso = Lasso()
pipe = make_pipeline(column_trans,scaler,lasso)
pipe.fit(x_train,y_train)
y_pred_lasso  = pipe.predict(x_test)
print(r2_score(y_test, y_pred_lasso))

# RIDGE
ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(x_train, y_train)
y_pred_ridge  = pipe.predict(x_test)
print(r2_score(y_test, y_pred_ridge))

import pickle

pickle.dump(pipe, open('RidgeModel.pkl','wb'))

print(data)

# # import pandas as pd
# # import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import LabelEncoder
# import pickle
#
# # Load the saved Ridge model
# with open('RidgeModel.pkl', 'rb') as f:
#     ridge_model = pickle.load(f)
#
# # Prepare new data sample
# data = {
#     'location': ['1st Block Jayanagar'],
#     'total_sqft': [2000.0],
#     'bath': [3.0],
#     'bhk': [3]
# }
# df = pd.DataFrame(data)
#
# # Apply necessary preprocessing steps
# # For example, encode categorical variable 'location' using LabelEncoder
# encoder = LabelEncoder()
# df['location'] = encoder.fit_transform(df['location'])
#
# # Make predictions
# predicted_prices = ridge_model.predict(df)
#
# # Print the predicted price
# print(predicted_prices)