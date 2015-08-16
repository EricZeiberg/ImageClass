import numpy as np
from PIL import Image
import os
import base64
from StringIO import StringIO
from sklearn.neighbors import KNeighborsClassifier

import utils

path = "/Coding/Data/gimpy-r-ball/"
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

image_files = import_data(path)
processed_data = process_data(image_files[0])
