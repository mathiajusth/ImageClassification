#Multilabel classification
import math
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import pandas as pd
import numpy as np
import argparse

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import random
random.seed(1234)

ap = argparse.ArgumentParser()

ap.add_argument("-data", "--data", required=True,
	help="dataset to be used")

ap.add_argument("-class", "--class", required = False,
	help="The class variable")

args = vars(ap.parse_args())

if args['data'] == "Pascal_SIFT_features.txt":
	X = pd.read_csv(args['data'],sep=" ", header=None)
else:
	X = pd.read_csv(args['data'],sep=",", header=None)


y = pd.read_csv(args['class'], sep =",", header='infer')

y = pd.concat([y['isA'].reset_index(), y['isP']], axis=1)

del y['index']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7.0, random_state=42)


from sklearn.decomposition import PCA
# Make an instance of the Model
"""
pca = PCA(.90)

pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
"""
#print(X_train.head())
#print(y_train.head())


# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(LogisticRegression())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report
#print("Binary relevance: ", accuracy_score(y_test,predictions))
#print("Hamming loss: ", hamming_loss(y_test, predictions))


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier

"""
if __name__ == '__main__':
	forest = RandomForestClassifier(n_estimators=100, random_state=1)
	multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
	predictions = multi_target_forest.fit(X_train, y_train).predict(X_test)
	acc = 0
	k=0
	for i in np.sum(y_test,axis=1):
		if i == np.sum(predictions,axis=1)[k]:
			acc=acc+1
		k=k+1

	 
	print("Sum accuracy: ", acc/(len(y_test)))

"""


#y = MultiLabelBinarizer().fit_transform(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#using classifier chains
from skmultilearn.problem_transform import ClassifierChain


# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(LogisticRegression())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
#print(predictions)
print("Log reg Classifier chains: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))

#In the multilabel case with binary label indicators:
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
#print(predictions)
print("Naive B Classifier chains: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))

#In the multilabel case with binary label indicators:
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")



classifier = ClassifierChain(svm.SVC(decision_function_shape='ovo'))

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
#print(predictions)
print("SVM Classifier chains: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))

#In the multilabel case with binary label indicators:
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")




# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(LogisticRegression())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

print("Log Reg Label Powerset: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")


classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

print("Naive B Label Powerset: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")




classifier = LabelPowerset(svm.SVC(decision_function_shape='ovo'))

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
#print(predictions)
print("SVM Classifier chains: ", accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))

#In the multilabel case with binary label indicators:
print("Hamming loss: ", hamming_loss(y_test, predictions))
print(" ")
