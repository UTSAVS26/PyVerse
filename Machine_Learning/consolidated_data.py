import cv2
import os
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(),'clean_data')
img_dir = os.path.join(os.getcwd(),'images')

print(os.getcwd())
image_data = []
labels = []

for i in os.listdir(img_dir):
    print(i)
    image = cv2.imread(os.path.join(img_dir,i))
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_data.append(image)
    labels.append(str(i).split("_")[0])
    
image_data = np.array(image_data)    
labels = np.array(labels) 
print(image_data)
print(labels)

import matplotlib.pyplot as plt
plt.imshow(image_data[300],cmap="gray")
plt.show()


with open(os.path.join(data_dir,"images.p"),'wb') as f:
    pickle.dump(image_data,f)
    
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)
    

