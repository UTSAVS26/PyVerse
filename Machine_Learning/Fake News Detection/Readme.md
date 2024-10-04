# PROJECT TITLE: Fake News Detector

## 🎯 Goal
The main goal of this project is to help users determine the authenticity of news articles by leveraging machine learning models. The purpose is to combat misinformation by providing real-time predictions on whether a news article is "Real" or "Fake."

## 🧵 Dataset
The dataset used in this project is sourced from  https://www.kaggle.com/competitions/fake-news/data?select=train.csv. 

## 🧾 Description
This web app utilizes machine learning algorithms to analyze news articles and classify them as either "Real" or "Fake." It features a user-friendly interface where users can input text, receive predictions, and provide feedback, contributing to the model's improvement over time.

## 🧮 What I had done!
1. **Data Collection**: Gathered and prepared a dataset of news articles for training the model.
2. **Model Training**: Developed and trained machine learning models to classify news articles.
3. **App Development**: Built a Flask web application to facilitate user interaction.
4. **Text Input**: Created a text input field for users to enter news articles.
5. **Prediction Mechanism**: Implemented real-time prediction of "Real" or "Fake" based on user input.
6. **Feedback System**: Enabled a feedback mechanism for users to report the accuracy of predictions.

## 🚀 Models Implemented
Model-1: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 12140, 182)        30222010  
                                                                 
 global_average_pooling1d (  (None, 182)               0         
 GlobalAveragePooling1D)                                         
                                                                 
 dense (Dense)               (None, 96)                17568     
                                                                 
 dense_1 (Dense)             (None, 24)                2328      
                                                                 
 dense_2 (Dense)             (None, 1)                 25        
                                                                 
=================================================================
Total params: 30241931 (115.36 MB)
Trainable params: 30241931 (115.36 MB)
Non-trainable params: 0 (0.00 Byte)

Model-2: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_3 (Embedding)     (None, 12140, 100)        16605500  
                                                                 
 simple_rnn (SimpleRNN)      (None, 10)                1110      
                                                                 
 dense_5 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 16606621 (63.35 MB)
Trainable params: 16606621 (63.35 MB)
Non-trainable params: 0 (0.00 Byte)

## 📚 Libraries Needed
- Flask
- Pandas
- NumPy
- Scikit-learn
- TensorFlow or Keras (depending on the model used)



## 📈 Performance of the Models based on the Accuracy Scores
### EVALUATION METRICS

The evaluation metrics I used to assess the models:

- Accuracy 
- Loss

It is shown using Confusion Matrix in the Images folder

### RESULTS
Results on Val dataset:
For Model-1:
Accuracy:96.11%
loss: 0.1350

For Model-2:
Accuracy:85.03%
loss: 0.1439


## 📢 Conclusion
The model-1 showed high validation accuracy of 96.11% and loss of 0.1350.Thus the model-1 worked fairly well identifying 2874 fake articles from a total of 3044.The first model performed better.The second model had good training accuracy but less test accuracy hinting towards overfitting.Maybe the key reason being in fake news it is important to capture overall sentiment better than individual word sentiment.

## ✒️ Your Signature
Pavitraa G
