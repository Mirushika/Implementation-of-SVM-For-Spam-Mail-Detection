# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import chardet for detecting the file encoding. 
2.Load the dataset using pd.read_csv() with the detected or appropriate encoding (Windows-1252 in this case). 
3.Fit the SVM model on the transformed training data (x_train and y_train). 
4.Calculate the accuracy of the model using metrics.accuracy_score(y_test, y_pred). 
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mirushika.T 
RegisterNumber: 24901203  
*/
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd 
data = pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```


## Output:
![SVM For Spam Mail Detection](sam.png)
```
{'encoding': 'Windows-1252', 'confidence': 0.7270322499829184, 'language': ''}

v1 v2 Unnamed: 2 Unnamed: 3 Unnamed: 4
0 ham Go until jurong point, crazy.. Available only ... NaN NaN NaN
1 ham Ok lar... Joking wif u oni... NaN NaN NaN
2 spam Free entry in 2 a wkly comp to win FA Cup fina... NaN NaN NaN
3 ham U dun say so early hor... U c already then say... NaN NaN NaN
4 ham Nah I don't think he goes to usf, he lives aro... NaN NaN NaN

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 v1 5572 non-null object
1 v2 5572 non-null object
2 Unnamed: 2 50 non-null object
3 Unnamed: 3 12 non-null object
4 Unnamed: 4 6 non-null object
dtypes: object(5)
memory usage: 217.8+ KB

v1 0
v2 0
Unnamed: 2 5522
Unnamed: 3 5560
Unnamed: 4 5566
dtype: int64

array(["Sorry, I'll call later", "Sorry, I'll call later",
"Sorry, I'll call later", ..., "Sorry, I'll call later",
"Sorry, I'll call later", "Sorry, I'll call later"], dtype=object)

0.003587443946188341
```

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
