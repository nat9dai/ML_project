from random import shuffle, choice
from PIL import Image
import os
import numpy as np
import matplotlib as plt
import random
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import pandas as pd
import cv2
import math
from random import randint



def load_rgb_data(IMAGE_DIRECTORY,IMAGE_SIZE, directory_depth=2, max_number_of_images = None, shuffle=True):
    '''
    The method will load all the rgb images from all subfolders recurively 
    '''
    print("Loading images...")
    data = []
    #labels=[]

    count=0
    for folder,directory_name,file_name in os.walk(IMAGE_DIRECTORY):
        #count=count+1
          
        if (len(file_name)!=0):
            #print (folder, directory_name, file_name)
            if (directory_depth!=0):
                folders_list =folder.split('/')[0:-1]
            else:
                folders_list =folder.split('/')[-1:]
            #print(folders_list)
            label = folders_list[-directory_depth]
            #print (label)
            for i in range(len(file_name)):
                image_name = file_name[i]
                image_path = os.path.join(folder, image_name)
                if ('.DS_Store' not in image_path):
                    #print(image_path)            
                    #try:
                    if True:
                        img = Image.open(image_path)
                        rgbimg = Image.new("RGB", img.size)
                        rgbimg.paste(img)
                        img=rgbimg

                        #print(np.array(img).shape)
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                        #print(np.array(img).shape)
                        data.append([np.array(img), label])
                        count = count +1
                        if(count%50==0):
                            print("loaded ",count, " images so far ...")  
                        if (max_number_of_images != None):
                            if (count==max_number_of_images):
                                images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
                                if (images.shape[2]>3): #sometime the image comes with RGBA with 4 channels
                                    images = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2BGR)
                                labels = np.array([i[1] for i in data])
                                return images,labels
                    #except Exception:
                    #    print("cannot load ", image_path)
                    #    pass
    print("number of images loaded: ",count)
    if (shuffle):
        print("shuffling data ...")
        random.shuffle(data)
        print ("dataset shuffled.")
        
    print ("convert images and labels into numpy array")  
    images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    labels = np.array([i[1] for i in data])
    
    print ("data shape:", images.shape)
    print ("labels shape:", labels.shape)
    print ("unique labels:", np.unique(labels))
    print ("loading dataset completed from path: ",IMAGE_DIRECTORY)
    
    return images,labels


def normalize_data(dataset):
  print("normalize data")
  dataset= dataset/255.0
  return dataset
     


def display_image(trainX, trainY, index=0):
  plt.imshow(trainX[index])
  print ("Label = " + str(np.squeeze(trainY[index])))
  print ("image shape: ",trainX[index].shape)

def display_one_image(one_image, its_label):
  plt.imshow(one_image)
  print ("Label = " + its_label)
  print ("image shape: ",one_image.shape)

def display_dataset_shape(X,Y):
  print("Shape of images: ", X.shape)
  print("Shape of labels: ", Y.shape)
  

def plot_sample_from_dataset(images, labels,rows=5, colums=5, width=8,height=8):

  plt.figure(figsize=(width,height))
  for i in range(rows*colums):
      plt.subplot(rows,colums,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(images[i], cmap=plt.cm.binary)
      plt.xlabel(labels[i])
  plt.show()

def display_dataset_folders(path):
  classes=os.listdir(path)
  classes.sort()
  print(classes)
  

def get_data_distribution(IMAGE_DIRECTORY, output_file=None,plot_stats=True):
    print("Loading images...")
    #list structure to collect the statistics
    stats=[]

    #get all image directories
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for diretcory_name in directories:
        print("Loading {0}".format(diretcory_name))
        images_file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, diretcory_name)))[2]
        print("we will load [", len(images_file_names), "] files from [",diretcory_name,"] class ..." )
        for i in range(len(images_file_names)):
          image_name = images_file_names[i]
          image_path = os.path.join(IMAGE_DIRECTORY, diretcory_name, image_name)
          #print(image_path)

          #the class is assumed to be equal to the directorty name
          label = diretcory_name 

          img = Image.open(image_path)
          #convert any image to RGB to make sure that it has three channels
          rgbimg = Image.new("RGB", img.size)
          rgbimg.paste(img)
          img=rgbimg
          
          #get the width and the height of the image in pixels
          width,height = img.size
          #get the size of the image in KB
          size_kb=os.stat(image_path).st_size/1000
          #add the size to a list of sizes to be 
          stats.append([label,os.path.basename(image_name),width,height,size_kb])

    if (output_file is not None):
      #convert the list into a dataframe to make it easy to save into a CSV
      stats_dataframe = pd.DataFrame(stats,columns=['Class','Filename','Width','Height','Size_in_KB'])
      stats_dataframe.to_csv(output_file,index=False)
      print("Stats collected and saved in .",output_file)
    else:
      print("Stats collected");


    return stats


def plot_dataset_distribution (stats, num_cols=5, width=10, height=5, histogram_bins = 10, histogram_range=[0, 1000], figure_padding=4):
  #convert the list into a dataframe
  stats_frame = pd.DataFrame(stats,columns=['Class','Filename','Width','Height','Size_in_KB'])

  #extract the datframe related to sizes only
  list_sizes=stats_frame['Size_in_KB']

  #get the number of classes in the dataset
  number_of_classes=stats_frame['Class'].nunique()
  print(number_of_classes, " classes found in the dataset")

  #create a list of (list of sizes) for each class of images
  #we start by the the sizes of all images in the dataset
  list_sizes_per_class=[list_sizes] 
  class_names=['whole dataset']
  print("Images of the whole dataset have an average size of ", list_sizes.mean())
  
  for c in stats_frame['Class'].unique():
    print("sizes of class [", c,"] have an average size of ", list_sizes.loc[stats_frame['Class']== c].mean())
    #then, we append the sizes of images of a particular class
    list_sizes_per_class.append(list_sizes.loc[stats_frame['Class'] == c])
    class_names.append(c)

      
  class_count_dict={}
  for c in stats_frame['Class'].unique():
    print("number of instances in class [", c,"] is ", stats_frame.loc[stats_frame['Class']== c].count()['Class'])
    #then, we append the sizes of images of a particular class
    class_count_dict[c]=stats_frame.loc[stats_frame['Class']== c].count()['Class']
    #list_sizes_per_class.append(list_sizes.loc[stats_frame['Class'] == c])
    #class_names.append(c)
    

    num_rows=math.ceil((number_of_classes+1)/num_cols)
  if (number_of_classes<num_cols):
    num_cols=number_of_classes+1
  fig,axes = plt.subplots(num_rows, num_cols, figsize=(width,height))
    

  fig.tight_layout(pad=figure_padding)
  class_count=0
  if (num_rows==1 or num_cols==1):
    for i in range(num_rows):
      for j in range(num_cols): 
        axes[j+i].hist(list_sizes_per_class[num_cols*i+j], bins = histogram_bins, range=histogram_range)
        axes[j+i].set_xlabel('Image size (in KB)', fontweight='bold')
        axes[i+j].set_title(class_names[j+i] + ' images ', fontweight='bold')
        class_count=class_count+1
        if (class_count==number_of_classes+1):
          break
  
  else:
    for i in range(num_rows):
      for j in range(num_cols): 
        axes[i,j].hist(list_sizes_per_class[num_cols*i+j], bins = histogram_bins, range=histogram_range)
        axes[i,j].set_xlabel('Image size (in KB)', fontweight='bold')
        axes[i,j].set_title(class_names[num_cols*i+j] + ' images ', fontweight='bold')
        class_count=class_count+1
        if (class_count==number_of_classes+1):
          break
    
    f=figure()
    print(class_count_dict)
    plt.bar(*zip(*class_count_dict.items()))
    
    for index, car_brand in enumerate(class_count_dict):
        plt.text(car_brand, class_count_dict[car_brand]+1, str(class_count_dict[car_brand]))
    #axes[1,3].set_xlabel(range(len(class_count_dict)), list(class_count_dict.keys()))
    plt.show()






