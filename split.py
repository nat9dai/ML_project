import h5py
import numpy as np
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical

from anis_koubaa_udemy_computer_vision_lib import *

from pathlib import Path

IMAGE_SIZE=524

#CHANGE THESE PATHS ACCORDING TO THE PATH IN YOUR SYSTEM
DATASET_PATH='dataset-realwaste-hierarchical'
H5DATASET_FOLDER='/home/nat/DL_in_practice/ML_project/h5-dataset-realwaste-flat/'
#HDF5_DATASET_PATH=H5DATASET_FOLDER+'realwaste-dataset-SIZE'+str(IMAGE_SIZE)+'.hdf5'
#HDF5_TARGET_DATASET_PATH=H5DATASET_FOLDER+'realwaste-dataset-SIZE'+str(IMAGE_SIZE)+'.hdf5.csv'
HDF5_DATASET_PATH=H5DATASET_FOLDER+'h5-dataset-realwaste-flatrealwaste-dataset-SIZE524.hdf5'
HDF5_TARGET_DATASET_PATH=H5DATASET_FOLDER+'h5-dataset-realwaste-flatrealwaste-dataset-SIZE524.hdf5.csv'

#read dataset
hf = h5py.File(HDF5_DATASET_PATH, "r")

class_label_string_length="S36"

#extract labels
labels_in_ascii = np.array(hf["labels"]).astype(class_label_string_length)
#print (labels)

data = np.array(hf["images"]).astype("f8")

#get labels in string format: decode from ASCII
labels = [n.decode('unicode_escape') for n in labels_in_ascii]
#print(asciiList)
print ('number of labels:',len(labels))

np.unique(labels)

#use this method if you want to select a sample of data uniformly distributed
def select_uniformly_distributed_data_sample(data,labels, class_dict, max_number_of_images=10):
    sample_data=[]
    sample_labels=[]
    for i,image in enumerate(data):
        
        label=labels[i]
        #print(label)
        #print(class_dict[label])
        if (class_dict[label]<max_number_of_images):
            #print(label)
            sample_data.append(image)
            sample_labels.append(label)
            class_dict[label]=class_dict[label]+1
    to_continue=False
    for x,y in class_dict.items():
        if y<max_number_of_images:
            to_continue==True
    #print(to_continue)
    if to_continue==False:
        return np.array(sample_data), np.array(sample_labels),class_dict
    
class_dict = {}
for car_class in np.unique(labels):
    class_dict[car_class]=0
    #print (class_dict)
print(class_dict)

#use this method if you want to use the whole dataset as is without balancing the classes
sample_data, sample_labels,class_dict = data,np.array(labels),class_dict

print(sample_data.shape)
print(sample_labels.shape)
print(class_dict)

plot_sample_from_dataset(sample_data, sample_labels,rows=6, colums=4, width=20,height=30)

test_split_ratio=0.1
(trainX, testX, trainLabels, testLabels) = train_test_split(sample_data, sample_labels,test_size=test_split_ratio, stratify=sample_labels, random_state=42)

# perform one-hot encoding on the labels
print ("sample of train labels: ", trainLabels[:4])
lb = LabelBinarizer()
train_binary_labels = lb.fit_transform(trainLabels)
print ("sample of train_binary_labels after Binarizer: \n", train_binary_labels[:4])


print ("sample of test labels: ", testLabels[:4])
test_binary_labels = lb.fit_transform(testLabels)
print ("sample of test_binary_labels after Binarizer: \n", test_binary_labels[:4])

trainY=train_binary_labels
testY=test_binary_labels

print("trainX.shape: ",trainX.shape)
print("trainY.shape: ",trainY.shape)
print("testX.shape: ",testX.shape)
print("testY.shape: ",testY.shape)

#print("testX.shape: ",testX.shape)
#print("testLabels.shape: ",np.array(testLabels).shape)

dev_test_ratio=0.5
(devX, testX, devLabels, testLabels) = train_test_split(testX, testLabels,test_size=dev_test_ratio, stratify=testLabels, random_state=42)

# perform one-hot encoding on the labels
print ("sample of dev labels: ", devLabels[:4])
lb = LabelBinarizer()
train_binary_labels = lb.fit_transform(devLabels)
print ("sample of dev_binary_labels after Binarizer: \n", train_binary_labels[:4])


print ("sample of test labels: ", testLabels[:4])
test_binary_labels = lb.fit_transform(testLabels)
print ("sample of test_binary_labels after Binarizer: \n", test_binary_labels[:4])

devY=train_binary_labels
testY=test_binary_labels

print("devX.shape: ",devX.shape)
print("devY.shape: ",devY.shape)
print("testX.shape: ",testX.shape)
print("testY.shape: ",testY.shape)

import h5py
hf=h5py.File(HDF5_TARGET_DATASET_PATH, 'w')

hf.create_dataset("trainX",
                  shape=trainX.shape,
                  maxshape=trainX.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=trainX)

hf.create_dataset("trainY",
                  shape=trainY.shape,
                  maxshape=trainY.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=trainY)

hf.create_dataset("devX",
                  shape=devX.shape,
                  maxshape=devX.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=devX)

hf.create_dataset("devY",
                  shape=devY.shape,
                  maxshape=devY.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=devY)

hf.create_dataset("testX",
                  shape=testX.shape,
                  maxshape=testX.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=testX)

hf.create_dataset("testY",
                  shape=testY.shape,
                  maxshape=testY.shape,
                  compression="gzip",
                  compression_opts=9,
                  data=testY)

train_labels_ascii= [n.encode('unicode_escape') for n in trainLabels]
dev_labels_ascii= [n.encode('unicode_escape') for n in devLabels]
test_labels_ascii= [n.encode('unicode_escape') for n in testLabels]

#print(train_labels_ascii)

hf.create_dataset("trainLabels",
            shape=np.array(trainLabels).shape,
            compression="gzip",
            compression_opts=9,
            data=train_labels_ascii,dtype=class_label_string_length
            )

hf.create_dataset("devLabels",
            shape=np.array(devLabels).shape,
            compression="gzip",
            compression_opts=9,
            data=dev_labels_ascii,dtype=class_label_string_length
            )

hf.create_dataset("testLabels",
            shape=np.array(testLabels).shape,
            compression="gzip",
            compression_opts=9,
            data=test_labels_ascii,dtype=class_label_string_length
            )

hf.close()