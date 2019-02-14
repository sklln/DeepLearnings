# Always a good start to classification cause you don't have to have much prior information about the data, non-parametric
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv('\Work\Github DL\DeepLearnings\Data\Iris.csv')
df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)

plt.figure(figsize=(16,9))
#plt.scatter(points_1[:,0],points_1[:,1],color='red',label='Setosa')
#plt.scatter(points_2[:,0],points_2[:,1],color='black',label='Versicolor')
#plt.scatter(points_3[:,0],points_3[:,1],color='green',label='Virginica')
plt.legend()
plt.show()

y_pred_knn = []

def Prep_Data(df):
    from sklearn.model_selection import train_test_split
    X = df.values.tolist()
    Y = []
    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)
    return x_train, x_test, y_train, y_test

def getEuclideanDistances(test, train):
    dists = []
    for val in test:
        #dists.append(np.sqrt(sum(((val-a)**2 for a in train))))
        for train_Val in train:
            dists.append(distance.euclidean(val, train_Val))
    return dists

def perform_KNN(x_train, x_test, y_train, y_test, k):
    dists = getEuclideanDistances(x_train, x_test)
    distSort = sorted(dists)
    temp_target = y_train.tolist()
    votes = [0]*k
    #classes have to be sorted too not distSort (align these for voting)
    for i in range(k):
        votes[distSort[i]] += 1
    y_pred_knn.append(votes.index(max(votes)))
    print('Accuracy:', accuracy_score(y_test, y_pred_knn))
