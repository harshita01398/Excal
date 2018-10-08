import cv2
import numpy as np
from rectangle import edge_detect
import math

def deletelines(lines):
    for line in lines:
        for index,line2 in enumerate(lines):
            if abs(line[0][1] - line2[0][1]) < 10 or slope(line) > 10:
                #print (abs(line[0][1] - line2[0][1]))
                del lines[index]
    return lines



def slope(line):
    return math.degrees(math.atan(abs(line[0][1] - line[0][3])/abs(line[0][0] - line[0][2])))



def row_height(lines):
    avg_distance = 0
    count=0
    for idx,line in enumerate(lines):
        if idx == len(lines)-1:
            break

        dist = lines[idx+1][0][1] - line[0][1]
        if dist > 10:
            avg_distance += dist
            count += 1

    avg_distance /= count
    print(avg_distance)
    return int(avg_distance)



def detect(img):
    canny = edge_detect(img)
    sobel = cv2.Sobel(canny,cv2.CV_64F,0,1,ksize=5)
    row,col = sobel.shape

    sobel = cv2.erode(sobel,np.ones([1,int(col/20)]),iterations=1)
    sobel = cv2.dilate(sobel,np.ones([1,int(col/10)]),iterations=1)
    sobel = np.array(sobel,dtype=np.uint8)
    # print(sobel.shape,sobel.dtype)

    lines = cv2.HoughLinesP(sobel,1,np.pi/180,150,None,col/2,30)
    # print("Initial lines : " , len(lines))
    val = 0
    if lines is not None:
        lines = lines.tolist()
        lines.sort(key= lambda x : x[0][1])
        print(len(lines))
        lines = deletelines(lines)
        row_size = row_height(lines)

        for line in lines:

            if line[0][0] == line[0][2]:
                theta = 90
            else :
                theta = slope(line)
            if theta < 10:
                # cv2.line(img,(line[0][0],line[0][1]),(line[0][2],line[0][3]),(0,255,0),2)
                val+=1
        print(val)


        cv2.imshow("Sobel",sobel)
        cv2.imshow("Canny",canny)
        cv2.waitKey(0)

        return lines,row_size
