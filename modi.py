import cv2
import numpy as np
import math


def nothing(x) :
    
    pass


def delete_horizontal_lines(lines):
    
    for line in lines:
        
        for index,line2 in enumerate(lines):
            
            if abs(line[0][1] - line2[0][1]) < 20:
            
                #print (abs(line[0][1] - line2[0][1]))
            
                del lines[index]
    
    return lines


def delete_vertical_lines(lines):
    
    for line in lines:
        
        for index,line2 in enumerate(lines):
            
            if abs(line[0][0] - line2[0][2]) < 5:
            
                #print (abs(line[0][1] - line2[0][1]))
            
                del lines[index]
    
    return lines


def create_roi(lines):
    
    for index,line in enumerate(lines) :
        
        if index == len(lines)-1:
        
            break
        
        roi = img[line[0][1]:lines[index+1][0][1],:,:]
        
        row,col,_ = roi.shape
        
        if row > 20:
        
            cv2.imshow("ROI",roi)
        
            cv2.waitKey(0)
        


def get_horizontal_lines(processed_image):
    
    
    sobel = cv2.Sobel(processed_image,cv2.CV_64F,0,1,ksize=5)
    
    sobel = cv2.dilate(sobel,np.ones([3,3]),iterations=1)
    
    sobel = np.array(sobel,dtype=np.uint8)
    
    row,col = sobel.shape
    
    lines = cv2.HoughLinesP(sobel,1,np.pi/180,150,None,col/2,30)
    
    #print(len(lines))
    
    val = 0
    
    if lines is not None:
    
        lines = lines.tolist()
    
        lines.sort(key= lambda x : x[0][1])
    
        #print(lines)
    
        lines = delete_horizontal_lines(lines)

        print(len(lines))

        for line in lines:

            # if abs(line[0][1] - line[0][3]) < 10:

            val +=1

            cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),5)

        print(val)

        create_roi(lines)


def get_vertical_lines(processed_image):

    cv2.imshow('recieved Image',processed_image)
    
    #laplacian = cv2.Laplacian(processed_image,cv2.CV_64F)

    blur = cv2.bilateralFilter(processed_image,11,75,75)
    
    cv2.imshow('blur',blur)
    
    canny = cv2.Canny(blur,100,200)
    
    sobel = cv2.Sobel(canny,cv2.CV_64F,1,0,ksize=5)

    cv2.imshow('sobelx',sobel)


    #dilate = cv2.dilate(sobel,np.ones([3,3]),iterations=1)

    #cv2.imshow('dilatex',dilate)
    
    #dilate = np.array(dilate,dtype=np.uint8)

    sobel = np.array(sobel,dtype=np.uint8)

    sobel = cv2.Canny(sobel,100,200)
    
    #cv2.imshow('sobel_int',sobel)

    window1 = cv2.namedWindow('trackbar1',cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('thresh1','trackbar1',0,1000,nothing)

    #window2 = cv2.namedWindow('trackbar2',cv2.WINDOW_AUTOSIZE)

    #cv2.createTrackbar('thresh2','trackbar2',0,20,nothing)

    while(1) :

        val1 = cv2.getTrackbarPos('thresh1','trackbar1')

        #val2 = cv2.getTrackbarPos('thresh2','trackbar2')

        lines = cv2.HoughLinesP(sobel,1,np.pi/180,25,None,val1,20).tolist()

        if lines is not None :

            lines.sort(key = lambda x : x[0][0])

            lines = delete_vertical_lines(lines)

            img = cv2.imread('att.png')

            for line in lines:

                cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,0,255),5)

            cv2.imshow('image',img)
            
        if cv2.waitKey(1) == 27 :
            break



def preprocess_image(img):

    cv2.GaussianBlur(img,(15,15),1)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img,100,200)

    blur = cv2.GaussianBlur(canny,(15,15),1)
    
    return blur


def read_image():
    
    img =  cv2.imread("att.png")
    
    return img


img = read_image()

processed_image = preprocess_image(img)

#get_horizontal_lines(processed_image)

get_vertical_lines(img)

#cv2.imshow("Original",img)

#cv2.imshow("Sobel",sobel)

#cv2.imshow("Canny",canny)

#cv2.imshow("Erode",erode)

cv2.waitKey(0)

cv2.destroyAllWindows()
