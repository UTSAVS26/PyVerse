##Twitter Sentiment Analysis 


## Problem Statement

The ability to analyze user sentiment through tweets and comments can provide significant value to companies during __product launches__. By understanding customer behavior and incorporating __sentiment analysis__, companies can gain insights from user feedback. This empowers them to make informed decisions, take necessary actions, and improve overall __revenue__ by addressing customer concerns and making targeted improvements accordingly.

## Machine Learning and Data Science

Our approach involves utilizing machine learning techniques and text extraction to predict the sentiment of a given text, determining whether it is positive or negative. Initially, we will analyze the text and examine the various words present within it. Once we have a comprehensive understanding of the text, we will proceed with the machine learning analysis, employing deep neural networks. The output from this analysis will be utilized in subsequent machine learning operations to generate predictions regarding the sentiment of the text, specifically determining whether it is positive or negative.

## Natural Language Processing (NLP)
We would be using the __natural language processing__ that is required when doing the machine learning analysis. Performing the natural language processing ensures that the words that are present are converted into mathematical vectors that are used for different machine learning models for prediction. Once the mathematical vectors are converted into different vectors, they are given for the machine learning models for prediction respectively. Therefore, with the features that are present in the text along with some newly created features, the machine learning, and deep learning models would be using those techniques and ensures that they are getting the best outputs respectively. 

## Vectorizers 

It is important to use vectorizers that are important for machine learning. Therefore, a given text which is in the form of a string is converted into a vectorial representation which is what is being used by machine learning models for prediction. Below are some of the vectorizers that were used in the process of converting a text into a mathematical representation. 

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## Machine Learning Models

In this project, there was only one ML model used to get the prediction of the sentiment of tweets. Below is the model that was used for the task of prediction.

* [__Deep Neural Networks__](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

## Exploratory Data Analysis (EDA)

After performing __exploratory data analysis__, it could be seen based on the results that there is a comparatively more number of neutral sentences compared to either positive or negative sentiments. With the use of __word clouds__, it could be seen that words such as good, awesome, and great were used most frequently. On the contrary, it could be seen for the negative __word cloud__ that words such as hate, sorry and sad were used most frequently. 

We have an image depicting a dataframe and a list of features. We will utilize the 'text' feature as input and consider the 'sentiment' feature as our target variable. Our goal is to predict the likelihood of a text being categorized as positive, negative, or neutral.

<img src = "https://github.com/aviralgarg05/Twitter-Sentiment-Analysis/blob/main/images/Input%20Data.jpg"/>

The countplot below illustrates that the majority of texts are classified as neutral sentiment, while the count of negative and positive texts is comparatively lower. This indicates a higher prevalence of neutral sentiments in the dataset.

<img src = "https://github.com/aviralgarg05/Twitter-Sentiment-Analysis/blob/main/images/Sentiment%20countplot.jpg"/>

Wordcloud gives a good representation by the presence of words based on their size. In other words, more frequent words appear in higher size as compared to others. Words such as "thank" and "day" are used most often in the positive tweets. 

<img src = "https://github.com/aviralgarg05/Twitter-Sentiment-Analysis/blob/main/images/Positive%20wordcloud.jpg"/>

The wordcloud provided showcases negative tweets within the dataset. Notably, recurring words like "hate" and "sad" are prevalent, indicating their significance in identifying negative sentiment.

<img src = "https://github.com/aviralgarg05/Twitter-Sentiment-Analysis/blob/main/images/Negative%20wordcloud.jpg"/>

## Hyperparameter Tuning 

In our project, after gaining a comprehensive understanding of various machine learning models, we will proceed with hyperparameter tuning. This crucial step aims to select optimal hyperparameters that can yield the best results for each specific model. By carefully selecting these hyperparameters, we can enhance the accuracy and performance of our machine learning models. Our objective is to explore and grasp the influence of different hyperparameters on the models and how they impact the outcomes for various problem statements. Ultimately, we aim to apply these optimized machine learning models in a production environment, leveraging their capabilities to achieve the desired results.

## Results 

The observed discrepancy between the training loss and the test loss suggests the presence of overfitting in the data. Despite this, the model could still be utilized for predictions, considering its potential ability to generalize well on unseen test data, despite its exceptionally strong performance on the training data.

<img src = "https://github.com/aviralgarg05/Twitter-Sentiment-Analysis/blob/main/images/Model%20Performance.jpg"/>

That's it, you should be able to read the code now. Thanks. 

