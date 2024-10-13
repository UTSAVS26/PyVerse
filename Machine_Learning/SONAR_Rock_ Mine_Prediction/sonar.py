import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('sonar_data.csv')
data.shape()
data.info()
data.isnull().sum()
data.describe()
data.columns
sns.countplot(data[60])
data,groupby(60).mean()
x=data.drop(60,axis=1)
y=data[60]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
lr= LogisticRegression()
lr.fit(x_train,y_train)
y_pred1=lr.predict(x_test)
accuracy_score(y_test,y_pred1)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred2=knn.predict(x_test)
accuracy_score(y_test,y_pred2)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred3=rf.predict(x_test)
accuracy_score(y_test,y_pred3)
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
for i in ramge(len(x_train)):
  sgd.partial_fit(x_train[i:i+1],y_train[i:i+1],classes=['R','M'])
score=sgd.score(x_test,y_test)
print(score)
