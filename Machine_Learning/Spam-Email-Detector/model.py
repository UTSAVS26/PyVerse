#Import Dependencies
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#Load the dataset and split into train_test split
df=pd.read_csv("spambase.csv")
X=df.drop(columns='spam',axis=1)
y=df['spam']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
#Train the logistic Regression model
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipeline.fit(X_train, y_train)

#Model Prediction for train data
X_train_predict=pipeline.predict(X_train)
X_train_accuracy=accuracy_score(X_train_predict,y_train)
print("Train_Accuracy:",X_train_accuracy)
#Model Prediction for test data
X_test_predict=pipeline.predict(X_test)
X_test_accuracy=accuracy_score(X_test_predict,y_test)
print("Test Accuracy:",X_test_accuracy)
#Finding Precision Score
precision=precision_score(y_test,X_test_predict)
print("Precision Score:",precision)
#Finding Recall Score
recall=recall_score(y_test,X_test_predict)
print("Recall Score:",recall)
#Finding f1 score
f1=f1_score(y_test,X_test_predict)
print("F1 Score:",f1)
#Confusion Matrix
cm=confusion_matrix(y_test,X_test_predict)
sns.heatmap(cm,annot=True)
plt.title("Confusion Matrix")
plt.show()
