import pandas as pd
import scipy.stats as stats
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
data=pd.read_csv(r'C:\Users\PC\Desktop\heart.csv' )
dataset=data.to_numpy()

dataset=np.array(dataset)
class_instances=dataset[:,:-1]
class_target=dataset[:,-1]
class_instances =stats.zscore(class_instances,axis=0)
X_train, X_test, Y_train, Y_test=train_test_split(class_instances,class_target,test_size = 0.4,random_state = 30)
standarddev_train=np.std(X_train)
averagetrain=np.average(X_train)                    
X_train=(X_train - np.average(X_train)) / (np.std(X_train))
X_test=(X_test - averagetrain) / standarddev_train

neigh = KNeighborsClassifier(n_neighbors=50)
#neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, Y_train) 


print("predict_proba",neigh.predict_proba(X_test))
peredict_target=neigh.predict( X_test)
re_predict=peredict_target.reshape(1,-1)
re_test=Y_test.reshape(1,-1)
print("actual_lable:",re_test)
print("\n")
print("predict_lable:",re_predict)
pre= precision_score(Y_test,peredict_target)
print("precision_score :",pre)
f1_score=f1_score(Y_test, peredict_target)
print("f1_score : ",f1_score)
recall_score=recall_score(Y_test, peredict_target)
print("recall_score :",recall_score)
acc = accuracy_score(Y_test, peredict_target)
print("acc :",acc)






