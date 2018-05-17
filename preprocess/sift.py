import argparse as ap
import cv2
from imutils import paths
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)  # Listing the train_path directory


# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []  # Inilialising the list
image_classes = []  # Inilialising the list
class_id = 0

for training_name in training_names:  # Iterate over the training_names list
    dir = os.path.join(train_path, training_name)
    image_paths.append(dir)

#print(image_paths[0])

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

#(kps, descs) = sift.detectAndCompute(gray, None)



# List where all the descriptors are stored
des_list = []
# Reading the image and calculating the features and corresponding descriptors
i=1
for image_path in image_paths:
    im = cv2.imread(image_path)
    (kpts, des) = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))  # Appending all the descriptors into the single list
    print("Reading image: ", i)
    i=i+1


#print(des_list)

# Stack all the descriptors vertically in a numpy array
i=1
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors
    print("Progress: ", i)
    i=i+1

# Perform k-means clustering
k = 500  # Number of clusters
voc, variance = kmeans(descriptors, k, 1)  # Perform Kmeans with default values

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
    print("Extracting features...")

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# Calculating the number of occurrences
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Giving weight to one that occurs more frequently

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)  # Scaling the visual words for better Prediction

np.savetxt("COCO_SIFT_features.txt",im_features)
