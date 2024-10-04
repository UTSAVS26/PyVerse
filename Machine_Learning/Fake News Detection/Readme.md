# Fake News Detection in Python
This project is a Fake News Detection system using NLP and Logistic regression to make the prediction, on whether the news is fake or not. 

# Libraries used : 
- Numpy
- Pandas
- Matplotlib
- Seaborn
- re
- nltk
- sklearn

# Steps being followed : 
- Import libraries
- Load the dataset
- Data Processing
- Stemming
- Convert text to numerical data
- Splitting data into test and train
- Logistic Regression model
- Evaluate

# Included Gradio as User Interface
# How to use it:
- Run all the cells
- Download all the necessary libraries
- Get the model in your desired format(pkl,h5,keras etc) or in the given format  
- At  final cell get the output as gradio interface
- Insert your news in the input section
- Get the prediction from model as real or fake with score
- If score greater than 0.7 then it is more likely to be fake news
- Also  get the confidence score
