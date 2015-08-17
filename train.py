import numpy as np
from PIL import Image
import os
import base64
from StringIO import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA

import utils

path = "/Coding/Data/gimpy-r-ball/"
"""
The n_components variable should be equal to the factorial of the number of letters each captcha has. 
"""
pca = RandomizedPCA(n_components=24)
knn = KNeighborsClassifier()
def img_to_matrix(filename):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    if (path.endswith("/")):
        img = Image.open(path + filename)
    else:
        img = Image.open(path + '/' + filename)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def import_data(folder):
    """
    Imports data from a folder. More info on this function in the README
    """
    images = []
    labels = []
    print "Reading training data from file"
    for f in utils.listdir_hidden(folder):
        if (len(f.split(".")) == 3):
            images.append(f)
        else:
            if (folder.endswith("/")):
                label = open(folder + f, 'rt')
            else:
                label = open(folder + "/" + f, 'rt')
            labels.append(label.read().rstrip())
    print "Read " + str(len(images)) + " images and " + str(len(labels)) + " labels. (These two values should be the same)"
    return [images, labels]

def process_data(image_array):
    data = []
    print "Transforming images to matrix of RGB pixels and flattening"
    for image in image_array:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)
    return data

def fit_data(data):
    print "Running RandomizedPCA filter on dataset to reduce dimentions"
    train_x = pca.fit_transform(data)
    return train_x

def string_to_img(img_path):
    print "called string_to_img"
    img = Image.open(img_path)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return pca.transform(img_wide[0])

def classify_image(data):
    print "called classify_image"
    preds = knn.predict(data)
    return preds

image_files = import_data(path)
processed_data = process_data(image_files[0])
data_pca = fit_data(processed_data)
print "Training data on KNN"
fit = knn.fit(data_pca, image_files[1])
print fit
new_img = string_to_img("/Users/ericzeiberg/Desktop/captcha.jpg")
pred = classify_image(new_img)
print pred
