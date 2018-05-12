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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


df = pd.read_csv('C:/Users/Omistaja/Documents/VaihdonKurssit/MachineLearningAndStuff/Results/FeaturesXception.csv', header='infer')
#print(df['class'])


for i in range(len(df['class'])):
	if (df['class'][i] == "cat"):
		df.loc[i,'class'] = 1

	elif(df['class'][i] == "clock"):
		df.loc[i,'class'] = 2

	elif(df['class'][i] == "dog"):
		df.loc[i,'class'] = 3

	elif(df['class'][i] == "giraffe"):
		df.loc[i,'class'] = 4

	elif(df['class'][i] == "pizza"):
		df.loc[i,'class'] = 5

	elif(df['class'][i] == "skis"):
		df.loc[i,'class'] = 6

	elif(df['class'][i] == "surfboard"):
		df.loc[i,'class'] = 7

	elif(df['class'][i] == "tennis racket"):
		df.loc[i,'class'] = 8

	elif(df['class'][i] == "toilet"):
		df.loc[i,'class'] = 9

	elif(df['class'][i] == "traffic light"):
		df.loc[i,'class'] = 10	
	


#print(df['class'])			


from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( df, df['class'], test_size=1/7.0, random_state=0)

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
