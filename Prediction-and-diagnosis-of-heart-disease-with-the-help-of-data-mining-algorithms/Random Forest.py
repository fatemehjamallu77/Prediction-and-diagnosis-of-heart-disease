import pandas as pd
import numpy as np 
#import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

data=pd.read_csv(r'C:\Users\PC\Desktop\heart.csv')
dataset=data.to_numpy()
dataset=np.array(dataset)
class_target=dataset[:,-1]
class_instances=dataset[:,:-1]
print(class_target)
print(class_instances)
#class_instances=stats.zscore(class_instances, axis=0)
X_train, X_test, Y_train, Y_test=train_test_split(class_instances,class_target,test_size = 0.4,random_state = 30)  

standarddev_train=np.std(X_train)
averagetrain=np.average(X_train)                             #نرمال
X_train=(X_train - np.average(X_train)) / (np.std(X_train))
X_test=(X_test - averagetrain) / standarddev_train

clf = RandomForestClassifier(n_estimators = 100)
clf = clf.fit(X_train,Y_train)
print("predict_proba",clf.predict_proba(class_instances[:1]))
peredict_target=clf.predict( X_test)
re_predict=peredict_target.reshape(1,-1)
re_test=Y_test.reshape(1,-1)
print("actual_lable:",re_test)
print("\n")
print("predict_lable:",re_predict)


cm = confusion_matrix(Y_test, peredict_target)
print('confusion_matrix:')
print(cm)
pre= precision_score(Y_test,peredict_target)
print("precision_score :",pre)
f1_score=f1_score(Y_test, peredict_target)
print("f1_score : ",f1_score)
recall_score=recall_score(Y_test, peredict_target)
print("recall_score :",recall_score)
acc = accuracy_score(Y_test, peredict_target)
print("acc :",acc)
data.keys()












