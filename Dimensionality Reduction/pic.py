#import tensorflow as tf
import math
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()

ap = argparse.ArgumentParser()

ap.add_argument("-data", "--data", required=True,
	help="dataset to be used")

ap.add_argument("-class", "--class", required = False,
	help="The class variable")

args = vars(ap.parse_args())




classVar = pd.read_csv(args['class'], header='infer')

features = pd.read_csv(args['data'],sep=" ", header=None)
#print(features.head(n=5))
df = pd.concat([classVar['x'].reset_index(), features], axis=1)

#print(df)


for i in range(len(df['x'])):
	if (df['x'][i] == "cat"):
		df.loc[i,'x'] = 1

	elif(df['x'][i] == "clock"):
		df.loc[i,'x'] = 2

	elif(df['x'][i] == "dog"):
		df.loc[i,'x'] = 3

	elif(df['x'][i] == "giraffe"):
		df.loc[i,'x'] = 4

	elif(df['x'][i] == "pizza"):
		df.loc[i,'x'] = 5

	elif(df['x'][i] == "skis"):
		df.loc[i,'x'] = 6

	elif(df['x'][i] == "surfboard"):
		df.loc[i,'x'] = 7

	elif(df['x'][i] == "tennis racket"):
		df.loc[i,'x'] = 8

	elif(df['x'][i] == "toilet"):
		df.loc[i,'x'] = 9

	elif(df['x'][i] == "traffic light"):
		df.loc[i,'x'] = 10	
	


del df['index']


from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(df, df['x'], test_size=1/7.0, random_state=0)

#print(df.head(n=5))

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)




"""
pca = PCA(0.2)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
# Predict for One Observation (image)
preds = logisticRegr.predict(test_img)
#print(pd.crosstab(test_lbl, preds, rownames=['Actual classes'], colnames=['Predicted classes']))
#print("Logistic Regression: ", logisticRegr.score(test_img, test_lbl))
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train_img,train_lbl)

print(logisticRegr.score(test_img, test_lbl))
print(clf.score(test_img, test_lbl))
"""

i=.2
pcas = []
accuracyLog = []
accuracyRf = []
accuracySvm = []
hamLoss = []
#Looping for image
while (i<=.99):
	pcas.append(i)
	pca = PCA(i)
	pca.fit(train_img)
	train_img_after_pca = pca.transform(train_img)
	test_img_after_pca = pca.transform(test_img)
	# all parameters not specified are set to their defaults
	# default solver is incredibly slow which is why it was changed to 'lbfgs'
	logisticRegr = LogisticRegression(solver = 'lbfgs')
	logisticRegr.fit(train_img_after_pca, train_lbl)
	# Predict for One Observation (image)
	#preds = logisticRegr.predict(test_img_after_pca)
	#print(pd.crosstab(test_lbl, preds, rownames=['Actual classes'], colnames=['Predicted classes']))
	#print("Logistic Regression: ", logisticRegr.score(test_img, test_lbl))

	accuracyLog.append(logisticRegr.score(test_img_after_pca, test_lbl))
	#print("Cross validation: ", cross_val_score(logisticRegr, features, df['x'], scoring='accuracy',cv=5))
	#print("Hamming loss: ", hamming_loss(test_lbl, preds))
	#hamLoss.append(hamming_loss(test_lbl, preds))

	#Random Forest
	# Create a random forest Classifier. By convention, clf means 'Classifier'
	clf = RandomForestClassifier(n_jobs=2, random_state=0)
	clf.fit(train_img_after_pca,train_lbl)
	#preds =clf.predict(test_img)
	#print(pd.crosstab(test_lbl, preds, rownames=['Actual classes'], colnames=['Predicted classes']))
	accuracyRf.append(clf.score(test_img_after_pca, test_lbl))
	#print("Cross validation: ", cross_val_score(clf, features, df['x'], scoring='accuracy',cv=5))

	#Support Vector Machine
	svmclf = svm.SVC(decision_function_shape='ovo')
	svmclf.fit(train_img_after_pca,train_lbl)
	accuracySvm.append(svmclf.score(test_img_after_pca, test_lbl))
	#preds =svmclf.predict(test_img)
	#print(pd.crosstab(test_lbl, preds, rownames=['Actual classes'], colnames=['Predicted classes']))
	#print("Default Support Vector Machines: ", svmclf.score(test_img, test_lbl))
	#print("Cross validation:", cross_val_score(svmclf, features, df['x'], scoring='accuracy',cv=5))
	i=i+0.01

#print(accuracyLog)

plt.plot(pcas,accuracyLog,'r',pcas,accuracyRf,'m',pcas,accuracySvm,'b')

plt.xlabel('Variance explained in PCA')
plt.ylabel('Accuracy')
plt.title('Accuracies with different PCA')
plt.savefig("test.jpg")
plt.show()
