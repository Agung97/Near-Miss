# import modul yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import missingno as msno 
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

#Cek rasio label data
df['test_outcome'].value_counts()/df.shape[0]

#Split data dengan metode stratified
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, stratify=df.test_outcome)
X_train = train.drop(['test_outcome'], axis=1)
y_train = train['test_outcome']
X_test = test.drop(['test_outcome'], axis=1)
y_test = test['test_outcome']
train.pivot_table(index='test_outcome', aggfunc='size').plot(kind='bar', title='Verify that class distribution in train is same as input data')
test.pivot_table(index='test_outcome', aggfunc='size').plot(kind='bar', title='Verify that class distribution in train is same as input data')

#---------------------------------------------------------------------------------------------------------------------------------------------------#


# apply near miss
from imblearn.under_sampling import NearMiss
nr = NearMiss()
  
#Undersampling pada data training
print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train == 0)))

X_train_miss, y_train_miss = nr.fit_resample(X_train, y_train.ravel())
  
print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape))
  
print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 0)))

#Undersampling pada data testing
print("Before Undersampling, counts of label '1': {}".format(sum(y_test == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_test == 0)))
  
X_test_miss, y_test_miss = nr.fit_resample(X_test, y_test.ravel())
  
print('After Undersampling, the shape of test_X: {}'.format(X_test_miss.shape))
print('After Undersampling, the shape of test_y: {} \n'.format(y_test_miss.shape))
  
print("After Undersampling, counts of label '1': {}".format(sum(y_test_miss == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_test_miss == 0)))

#----------------------------------------------------------------------------------------------------------------------------#

#Cek distribusi label setelah undersampling
#Data training
pd.Series(y_train_miss).value_counts().plot(kind='bar', title='Class distribution after applying NearMiss', xlabel='test_outcome')
#Data testing
pd.Series(y_test_miss).value_counts().plot(kind='bar', title='Class distribution after applying NearMiss', xlabel='test_outcome')
