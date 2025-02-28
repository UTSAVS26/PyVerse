import numpy as np
import argparse
import cv2
import os

DIR=r"C:\Users\Your-Directory\PycharmProjects\Colourisation"
PROTOTXT=os.path.join(DIR,r"For Colorisation\colorization_deploy_v2.prototxt")
POINTS=os.path.join(DIR,r"For Colorisation\pts_in_hull.npy")
MODEL=os.path.join(DIR,r"For Colorisation\colorization_release_v2.caffemodel")

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",type=str, required=True,
                help="path to input black and white image")
args=vars(ap.parse_args())

print("Load model")
net=cv2.dnn.readNetFromCaffe(PROTOTXT,MODEL)
pts=np.load(POINTS)

class8=net.getLayerId("class8_ab")
conv8=net.getLayerId("conv8_313_rh")
pts=pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs=[pts.astype("float32")]
net.getLayer(conv8).blobs=[np.full([1,313],2.606,dtype="float32")]

image=cv2.imread(args["image"])
scaled=image.astype("float32")/255.0
lab=cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

resized=cv2.resize(lab,(224,224))
L=cv2.split(resized)[0]
L-=50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0,:,:,:].transpose((1,2,0))

ab=cv2.resize(ab,(image.shape[1],image.shape[0]))

L=cv2.split(lab)[0]
colorized=np.concatenate((L[:,:,np.newaxis],ab),axis=2)

colorized=cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized=np.clip(colorized,0,1)

colorized=(255* colorized).astype("uint8")

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Colorized", cv2.WINDOW_NORMAL)

# Set the window sizes to desired dimensions (e.g., 600x400)
cv2.resizeWindow("Original", 800, 400)
cv2.resizeWindow("Colorized", 800, 400)

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Switch to the folder where this project is via command prompt
#Run this Command   python p1.py --image images/space4.jpg
