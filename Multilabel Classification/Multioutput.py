##Multioutput classifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import random
random.seed(1234)

ap = argparse.ArgumentParser()

ap.add_argument("-data", "--data", required=True,
	help="dataset to be used")

ap.add_argument("-class", "--class", required = False,
	help="The class variable")

ap.add_argument("-binary", "--binary", required = False,
	help="The class variable")

args = vars(ap.parse_args())

if args['data'] == "Pascal_SIFT_features.txt":
	X = pd.read_csv(args['data'],sep=" ", header=None)
else:
	X = pd.read_csv(args['data'],sep=",", header=None)


y = pd.read_csv(args['class'], sep =",", header='infer')

if args['binary'] == "True":
	y = pd.concat([y['isA'].reset_index(), y['isP']], axis=1)
else:
	y = pd.concat([y['aerop'].reset_index(), y['pers']], axis=1)



del y['index']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7.0, random_state=42)


from sklearn.decomposition import PCA
# Make an instance of the Model
"""
pca = PCA(.99)

pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
"""

from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':
	#forest = RandomForestClassifier(n_estimators=100, random_state=1)
	##Out of laziness, forest stands for Logistic Regression now
	forest = LogisticRegression()
	bayes = GaussianNB()
	svm = LinearSVC()
	multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
	multi_target_bayes = MultiOutputClassifier(bayes, n_jobs=-1)
	multi_target_svm = MultiOutputClassifier(svm, n_jobs=-1)

	predictionsF = multi_target_forest.fit(X_train, y_train).predict(X_test)
	predictionsB = multi_target_bayes.fit(X_train, y_train).predict(X_test)
	predictionsS = multi_target_svm.fit(X_train, y_train).predict(X_test)
	if args['binary'] == "True":
		print("Accuracy Random Forest: ", accuracy_score(y_test,predictionsF))
		print(classification_report(y_test,predictionsF))
		print("Hamming loss: ", hamming_loss(y_test, predictionsF))
		print("Accuracy Naive Bayes: ", accuracy_score(y_test,predictionsB))
		print(classification_report(y_test,predictionsB))
		print("Hamming loss: ", hamming_loss(y_test, predictionsB))
		print("Accuracy SVM: ", accuracy_score(y_test,predictionsS))
		print(classification_report(y_test,predictionsS))
		print("Hamming loss: ", hamming_loss(y_test, predictionsS))

	accF = 0
	accB = 0
	accS = 0
	k=0
	l=0
	j=0
	for i in np.sum(y_test,axis=1):
		if i == np.sum(predictionsF,axis=1)[k]:
			accF=accF+1
		k=k+1

	for i in np.sum(y_test,axis=1):
		if i == np.sum(predictionsB,axis=1)[l]:
			accB=accB+1
		l=l+1

	for i in np.sum(y_test,axis=1):
		if i == np.sum(predictionsS,axis=1)[j]:
			accS=accS+1
		j=j+1

	 
	print("Sum accuracy F: ", accF/(len(y_test)))
	print("Sum accuracy B: ", accB/(len(y_test)))
	print("Sum accuracy S: ", accS/(len(y_test)))
