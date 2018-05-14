#import tensorflow as tf
import math
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

ap = argparse.ArgumentParser()

ap.add_argument("-data", "--data", required=True,
	help="dataset to be used")

ap.add_argument("-class", "--class", required = False,
	help="The class variable")

args = vars(ap.parse_args())




classVar = pd.read_csv(args['class'], header='infer')

features = pd.read_csv(args['data'], header=None)
#print(features.head(n=5))
df = pd.concat([classVar['x'].reset_index(), features], axis=1)



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

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)

pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')

print(logisticRegr.fit(train_img, train_lbl))

# Predict for One Observation (image)
print(logisticRegr.predict(test_img[0].reshape(1,-1)))

# Predict for One Observation (image)
print(logisticRegr.predict(test_img[0:10]))

print(logisticRegr.score(test_img, test_lbl))


