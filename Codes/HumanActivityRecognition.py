'''
A work assignment from Deakin University, VIC 3216, Australia.
the paper work and data sets can be found here
https://drive.google.com/file/d/0BycCgbi2uOfqWjFPbkU5ZmQ2eVE/view?ts=59cceb6e
as far as now part 1 and 2 is done 
'''

'''
Part 1:
    * To get to know about different kind of human activity 
    * Human Activities are : WALKING ,WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
      there is 30 subjects performed these activities
    * in training set we have 7352 instances and the test is 2947
      number of features is: 561, each feature instance describe the current humanbody status in 
      3 dimensions representation with a lot of information and signals to process
    * This machine learning module is used to recognize the human activities and classify 
      them to the mentioned activities labels, maximum accuracy achieved is 0.894808279606 using KNeighborsClassifier
'''
#####################loading and training the module using KNeighborsClassifier########################
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# read train data ((check where is your data is stored))
X_train = pd.read_csv(sep=' ',
                      filepath_or_buffer='train/X_train.txt',
                      header=None,
                      skipinitialspace=True
                      )

# read test data
X_test = pd.read_csv(sep=' ',
                     filepath_or_buffer='test/X_test.txt',
                     header=None,
                     skipinitialspace=True
                     )

# read train target data as y
Y_train = pd.read_csv(
    filepath_or_buffer='train/Y_train.txt',
    header=None
)
Y_train = Y_train[0]
# read test target data as
Y_test = pd.read_csv(
    filepath_or_buffer='test/Y_test.txt',
    header=None
)
# read features names
features = pd.read_csv(sep=" ",
                       filepath_or_buffer="features.txt",
                       header=None
                       )
# select only the feature name
features = features.loc[:, 1]

# rename train data columns to given features names
X_train.columns = features

# rename target data
Y_train.columns = ["Activity_Label"]

neigh = KNeighborsClassifier(n_neighbors=50)

neigh.fit(X_train, Y_train)

preds = neigh.predict(X_test)

print accuracy_score(Y_test, preds)
#####################################################
'''
   Part 2 :
'''

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

myList = list(range(50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

#####################confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = [1, 2, 3, 4, 5, 6]
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
################################################
'''
    Part 3 :
'''
# Multiclass Logistic Regression with Elastic Net
