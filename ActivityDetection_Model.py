import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

#Setting Parameters
#winSize in seconds
#compDistance in seconds
#jumpThreshold in meters
##########
train_file_name = 'train_s3'
test_file_name = 'test_nexus4'
ext = '.csv'


#Importing dataset
dataset = pd.read_csv(train_file_name+ext)
test_dataset = pd.read_csv(test_file_name+ext)
X_train= dataset.iloc[: ,0:-1].values
Y_train= dataset.iloc[:,-1].values
X_test= test_dataset.iloc[:,0:-1].values
Y_test= test_dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('PCA: '+ str(explained_variance))

'''
from matplotlib.colors import ListedColormap
X_set , y_set = X_train , Y_train
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j , s=5)
plt.show()

'''

#Fitting classifier to the Training set(SVM)
Model = 'SVM(rbf) Kernel Trick'
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train,Y_train)

'''
#Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
Model = 'KNN'
classifier = KNeighborsClassifier(n_neighbors=5,metric = 'minkowski', p = 2)
classifier.fit(X_train,Y_train)
'''

#Predicting the test set results
Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
print(Model)
print('Confusion Matrix(Train):')
cm=confusion_matrix(Y_train,Y_pred_train)
print(cm)
print('Confusion Matrix(Test):')
cm=confusion_matrix(Y_test,Y_pred_test)
print(cm)
print('F1 Score:' + str(f1_score(Y_test,Y_pred_test)))
print('Accuracy(Train):' + str(accuracy_score(Y_train,Y_pred_train)))
print('Accuracy(Test):' + str(accuracy_score(Y_test,Y_pred_test)))







#Visualizing the dataset

#Train
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j , s = 10)
plt.title(Model+'(Train set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#Test
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j , s = 10)
plt.title(Model+ '(Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

