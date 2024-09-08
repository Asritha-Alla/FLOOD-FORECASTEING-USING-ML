import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

global accuracy, precision, recall, fscore
accuracy = []
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []
algorithm_name = []

le1 = LabelEncoder()
le2 = LabelEncoder()

dataset = pd.read_csv('Dataset/Flood.csv')
dataset.fillna(0, inplace = True)
dataset['SUBDIVISION'] = pd.Series(le1.fit_transform(dataset['SUBDIVISION'].astype(str)))
dataset['FLOODS'] = pd.Series(le2.fit_transform(dataset['FLOODS'].astype(str)))

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
X = normalize(X)
print(X)
print(Y)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])
    sp = cm[1,1]/(cm[1,0]+cm[1,1])
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    algorithm_name.append(algorithm)
    sensitivity.append(se)
    specificity.append(sp)
    print(str(a)+" "+str(p)+" "+str(r)+" "+str(f)+" "+str(se)+" "+str(sp))
    

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
predict = lr.predict(X_test)
calculateMetrics("Logistic Regression", predict, y_test)

svm_cls = SVC(kernel="rbf")
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
calculateMetrics("KNN", predict, y_test)

mlp = MLPClassifier(max_iter=2000)
mlp.fit(X, Y)
predict = mlp.predict(X_test)
calculateMetrics("MLP", predict, y_test)
















