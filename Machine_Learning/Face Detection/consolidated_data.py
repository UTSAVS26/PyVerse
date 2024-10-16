#Import the necessary libraries
import cv2
import os
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(),'clean_data') #Defines the path for the directory to store the processed image data
img_dir = os.path.join(os.getcwd(),'images') #Defines the path for the directory containing the original image files

print(os.getcwd()) # Prints the current working directory
image_data = []
labels = []

#Processing image data
for i in os.listdir(img_dir):
    print(i)
    image = cv2.imread(os.path.join(img_dir,i)) #Reads the image using cv2.imread
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_data.append(image) #Appends the processed image to image_data and the extracted label (first part of the filename) to labels
    labels.append(str(i).split("_")[0])
    
image_data = np.array(image_data)    #Converting to NumPy arrays
labels = np.array(labels) 
print(image_data)
print(labels)

#Visualization
import matplotlib.pyplot as plt
plt.imshow(image_data[300],cmap="gray")
plt.show()

#Saving data with pickle
with open(os.path.join(data_dir,"images.p"),'wb') as f:
    pickle.dump(image_data,f)
    
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)

#Make sure you have OpenCV, NumPy, pickle, and Matplotlib installed before running this code.
#You can modify the image size (100x100) based on your project requirements.
#This code assumes the filenames in img_dir follow a specific format (label_imagename.extension)
    

