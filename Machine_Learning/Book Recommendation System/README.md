# Book-Recommendation-System-Project
Objective

The main objective is to create a book recommendation system for users. Recommender systems are really critical in some industries as they can generate a huge amount of income when they are efficient or also be a way to stand out significantly from competitors.

Methods Used

Descriptive Statistics Data Visualization Machine Learning

Technologies

Python Pandas Numpy Matplotlib Seaborn Scikit-learn Surprise

Data

The Book-Crossing dataset comprises 3 files.

Users : Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL values.

Books : Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in the case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavors (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon website.

Ratings : Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

Project Description

EDA - Performed exploratory data analysis on numerical and categorical data.

Data Cleaning - Missing value imputation,Outlier Treaatment

Feature Selection - Used User-ID,ISBN and Books-Rating for model development.

Model development - Tried Popularity based model and Collaborative filtering (Both Memory based and Model based).

Needs of this project

data exploration data processing/cleaning recommendation system developer

Getting Started

Clone this repo (for help see this tutorial).

Raw Data

Users_data is being kept here within this repo.

Ratings_data is being kept here within this repo.

Books_data is being kept here

Complete notebook containing Data exploration/Data processing/transformation/model development is being kept here.
