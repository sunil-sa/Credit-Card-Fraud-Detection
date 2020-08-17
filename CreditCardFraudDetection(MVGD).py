import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
Data_csv = pd.read_csv('Creditcard.csv')
#Making a copy of the data file
data = Data_csv.copy(deep=True)
print(data.shape)
#Checking the number of missing values in each column
print(sum(data.isnull().sum()))
#Getting the data types of each column
print(data.dtypes)
print(data.head())
summary=data.describe()
#Getting the correlations betwee variables
correlation=data.corr()
#Visualizing the correlations
sns.heatmap(data.corr(), vmin=-1)
plt.show()
#We can see from the heatmap that the amount,time are slightly related with other variables
#Checking if the features are from gaussian distribution.
fig, axs = plt.subplots(6, 5, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_facecolor('xkcd:charcoal')
    ax.set_title(data.columns[i])
    sns.distplot(data.iloc[:, i], ax=ax, fit=norm,
                 color="cyan", fit_kws={"color": "black"})
    ax.set_xlabel('')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
plt.show()
#From the plot we can see that Time is not from gaussian distribution. After this analysysis it's decided 
#to drop the Time and Amount features.
df = data.drop(['Time','Amount','Class'],axis=1)
#Normalizing the data
Scaler = MinMaxScaler()
Scaled_df=Scaler.fit_transform(df)
df = pd.DataFrame(Scaled_df)
Class = data['Class']
df = pd.concat([df,Class],axis=1)
#0 represents valid transaction
#1 represents fraud transaction
print(df['Class'].value_counts())
#So as the data is skewed we can't use the supervised learning algorithms.
######### MULTIVARIATE GAUSSIAN DISTRIBUTION FOR ANOMALY DETECTION ###########
#Here we are assuming that the features are from gaussian distribution.
##splitting the data
def Splitdata(df):
    Normal=df[df['Class']==0]
    Fraud=df[df['Class']==1]
    Normal_count = Normal.shape[0]
    Fraud_count = Fraud.shape[0]
    #train_normal_count = int(Normal_count*0.6)
    cv_normal_count = int(Normal_count*0.2)
    cv_fraud_count = int(Fraud_count*0.5)
    train_normal_indices = list(range(cv_normal_count*2,Normal_count))
    cv_normal_indices = list(range(cv_normal_count))
    test_normal_indices = list(range(cv_normal_count,cv_normal_count*2))
    cv_fraud_indices = list(range(cv_fraud_count))
    test_fraud_indices = list(range(cv_fraud_count,Fraud_count))
    train_set = Normal.iloc[train_normal_indices,:]
    cv_normal_set=Normal.iloc[cv_normal_indices,:]
    test_normal_set = Normal.iloc[test_normal_indices,:]
    cv_fraud_set = Fraud.iloc[cv_fraud_indices,:]
    test_fraud_set = Fraud.iloc[test_fraud_indices,:]
    cv_set = pd.concat([cv_normal_set,cv_fraud_set],axis=0)
    test_set = pd.concat([test_normal_set,test_fraud_set],axis=0)
    Xtrain = train_set.drop(['Class'],axis=1)
    Xval = cv_set.drop(['Class'],axis=1)
    Yval = cv_set['Class']
    Xtest = test_set.drop(['Class'],axis=1)
    Ytest = test_set['Class']
    return Xtrain.to_numpy(), Xtest.to_numpy(), Xval.to_numpy(), Ytest.to_numpy(), Yval.to_numpy()
(Xtrain, Xtest, Xval, Ytest, Yval) = Splitdata(df)
#Finding the daussian parameters
def gaussian_params(X):
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    return mu, sigma
(mu, sigma) = gaussian_params(Xtrain)
# calculate gaussian pdf
p = multivariate_normal.pdf(Xtrain, mu, sigma)
pval = multivariate_normal.pdf(Xval, mu, sigma)
ptest = multivariate_normal.pdf(Xtest, mu, sigma)
def metrics(y, predictions):
    fp = np.sum(np.all([predictions == 1, y == 0], axis=0))
    tp = np.sum(np.all([predictions == 1, y == 1], axis=0))
    fn = np.sum(np.all([predictions == 0, y == 1], axis=0))
    if (tp + fp) > 0:
        precision = (tp / (tp + fp)) 
    else:
        precision = 0
    if (tp + fn) > 0:
        recall = (tp / (tp + fn)) 
    else:
        recall = 0
    if (precision + recall) > 0:
        F1 = (2 * precision * recall) / (precision +recall)  
    else:
        F1 = 0
    return precision, recall, F1,fp,fn
def FindEpsilon(yval, pval):
    e_values = pval
    bestF1 = 0
    bestEpsilon = 0
    for epsilon in e_values:
        predictions = pval < epsilon
        (precision, recall, F1,fp,fn) = metrics(yval, predictions)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
(epsilon, F1) = FindEpsilon(Yval, pval)
print("Best epsilon found:", epsilon)
print("Best F1 on cross validation set:", F1)
test_prediction = ptest<epsilon
(test_precision, test_recall, test_F1,fp,fn) = metrics(Ytest, test_prediction)
print("Test set F1 score:", test_F1)