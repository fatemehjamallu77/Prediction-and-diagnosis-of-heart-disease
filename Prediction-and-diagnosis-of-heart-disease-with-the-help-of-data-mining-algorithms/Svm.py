import pandas as pd
import numpy as np 
import scipy.stats as stats

from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data=pd.read_csv(r'C:\Users\PC\Desktop\heart.csv' )
dataset=data.to_numpy()

dataset=np.array(dataset)
class_instances=dataset[:,:-1]
class_target=dataset[:,-1]
class_instances =stats.zscore(class_instances,axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(class_instances, class_target, test_size = 0.4)
standarddev_train=np.std(X_train)
averagetrain=np.average(X_train)                            
X_train=(X_train - np.average(X_train)) / (np.std(X_train))
X_test=(X_test - averagetrain) / standarddev_train

clf = svm.SVC(C = 100, kernel = 'rbf', gamma = 0.001, probability = False)

clf = clf.fit(X_train, Y_train)

predicted = clf.predict(X_test)
pre= precision_score(Y_test, predicted)
print("precision_score :",pre)
f1_score=f1_score(Y_test, predicted)
print("f1_score : ",f1_score)
recall_score=recall_score(Y_test, predicted)
print("recall_score :",recall_score)
acc = accuracy_score(Y_test, predicted)
print("acc :",acc)



