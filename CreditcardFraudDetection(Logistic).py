#Credit Card fraud detection using Machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
Data_csv = pd.read_csv('Creditcard.csv')
#Making a copy of the data file
data = Data_csv.copy(deep=True)
print(data.shape)
#Checking the number of missing values in each column
print(data.isnull().sum())
#Getting the data types of each column
print(data.dtypes)
print(data.head())
summary=data.describe()
#0 represents valid transaction
#1 represents fraud transaction
print(data['Class'].value_counts())
columns_list=list(data.columns)
features=list(set(columns_list)-set(['Class']))
x=data[features].values
y=data['Class'].values
#Splitting the data as training and testing set
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#######  LOGISTIC REGRESSION FOR CLASSIFICATION #############################
logistic = LogisticRegression()
logistic.fit(train_x,train_y)
print(logistic.coef_)
print(logistic.intercept_)
#prediction on test data
prediction = logistic.predict(test_x)
#confusion matrix
confusion_mat = confusion_matrix(test_y,prediction)
print(confusion_mat)
#calculating the accuracy
accuracy = accuracy_score(test_y,prediction)
print(accuracy)
#printing the misclassified samples
print("Misclassified samples:",(test_y!=prediction).sum()) 
Recall=(confusion_mat[1][1])/(confusion_mat[1][1]+confusion_mat[1][0])
Precision=(confusion_mat[1][1])/(confusion_mat[1][1]+confusion_mat[0][1])
Fscore=(2*Recall*Precision)/(Recall+Precision)
print("F1 score on test set is:")
print(Fscore)
#F score is very low so logistic regression doesn't works in this case..let's switch to other model.