import pandas as pd
#import numpy as np 
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:/Users/pc/Desktop/heart.csv' )
dataset=data.to_numpy()
class_target=dataset[:,-1]
class_instances=dataset[:,:-1]
print(class_instances)
class_instances=stats.zscore(class_instances, axis=0)    #نرمال
print(class_instances)
model = DBSCAN(eps =3, min_samples =8)  
model = model.fit(class_instances) 
clusters = model.labels_  
print(clusters)

acc = accuracy_score(class_target, clusters)
print("acc :",acc)
print('silhouette:',silhouette_score(class_instances, clusters))

cm = confusion_matrix(class_target, clusters)
print('confusion_matrix:')
print(cm)

'''pre = precision_score(class_target, clusters)
print(" precision :",pre)
re = recall_score(class_target, clusters)
print(" recall:",re)
f1 = f1_score(class_target, clusters)
print(" f1_score:",f1)
'''
plt.hist(clusters)

two_d_data =class_instances [:, :2] 
model = DBSCAN(eps =3, min_samples = 8)
model = model.fit(two_d_data)

clusters = model.labels_

colors = clusters

plt.figure()

plt.scatter(two_d_data[:, 0], two_d_data[:, 1], c = colors, marker = 'x')

          







