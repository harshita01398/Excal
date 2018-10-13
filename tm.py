import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import cv2

"""
(X_train,y_train),(X_test,y_test) = mnist.load_data()

for i in range(0,60000):
    X_train[i] = cv2.Canny(X_train[i],100,200)
"""

"""
def nothing(x):
    pass
"""
img = cv2.imread('6.jpg')

#img = cv2.Canny(img,100,200)

#img = cv2.dilate(img,np.ones([3,3]),iterations = 1)

img = cv2.resize(img,(28,28))

"""
window1 = cv2.namedWindow('trackbar1',cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('thresh1','trackbar1',0,255,nothing)

window2 = cv2.namedWindow('trackbar2',cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('thresh2','trackbar2',0,255,nothing)


#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = cv2.blur(gray,(7,7))

#gray = cv2.GaussianBlur(gray,(7,7),0)



while(1) :

    val1 = cv2.getTrackbarPos('thresh1','trackbar1')

    val2 = cv2.getTrackbarPos('thresh2','trackbar2')
    
    canny = cv2.Canny(img,val1,val2)
    cv2.imshow('canny',canny)
    if (cv2.waitKey(1) == 27 ) :
        break


#canny = cv2.dilate(canny,np.ones([3,3]),iterations= 1)
"""

canny = cv2.Canny(img,100,200)

#canny = cv2.dilate(canny,np.ones([3,3]),iterations= 1)

cv2.imshow('2',canny)



rows,cols = canny.shape

print (canny.shape)
canny = canny.reshape(1, 1, 28, 28).astype('float32')

canny = canny/255   
print(canny)


cv2.waitKey(0)
cv2.destroyAllWindows()

model = load_model('2.h5')

weights=model.get_weights()

print (model.predict(canny,batch_size = 200,verbose = 2,steps = None))
#print (weights)
