import cv2
import numpy as np
import detectLines
import roi
import rectangle


img =  rectangle.resize(cv2.imread(input("Enter the name of the image : ")))

# lines = detectLines.detect(img)
# roi.create_row(img,lines)
rect_img = rectangle.draw(img)
lines,row_size = detectLines.detect(rect_img)
roi.create_row(rect_img,lines,row_size)


cv2.imshow("Rectangle",rect_img)
cv2.imshow("Original",img)


cv2.waitKey(0)
cv2.destroyAllWindows()
