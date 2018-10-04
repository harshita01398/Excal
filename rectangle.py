import cv2
import numpy as np

def draw(in_im,canny):

	cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

	img, c, h = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cMax = max(c, key = cv2.contourArea)
	# print(cMax)

	x,y,w,h = cv2.boundingRect(cMax)
	cv2.rectangle(in_im,(x,y),(x+w,y+h),(0,0,255),2)

	cv2.drawContours(in_im, cMax, -1, (255, 0, 0), 1)

	roi = in_im[y:y+h,x:x+w]
	return roi

# extLeft = tuple(cMax[cMax[:, :, 0].argmin()][0])
# extRight = tuple(cMax[cMax[:, :, 0].argmax()][0])
# extTop = tuple(cMax[cMax[:, :, 1].argmin()][0])
# extBot = tuple(cMax[cMax[:, :, 1].argmax()][0])
# cv2.circle(in_im, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extRight, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extTop, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extBot, 8, (0, 0, 255), -1)
# print(extLeft,extRight,extTop,extBot)
# cmaxx =0
# cmaxy = 0
# cminx = 10000
# cminy = 10000
# for cnt in cMax:
# 	# print(cnt[0])
# 	cx,cy = cnt[0]
# 	cminx = min(cminx,cx)
# 	cmaxx = max(cmaxx,cx)
# 	cminy = min(cminy,cy)
# 	cmaxy = max(cmaxy,cy)
# print(cminx,cminy)
# print(cmaxx,cmaxy)

# cv2.line(in_im,(cminx,cminy),(cminx,cmaxy),(255,0,0),5)
# cv2.drawContours(in_im, [cMax], cMax[-1], (255, 0, 0), 3)
# print(cMax[0],cMax[-1])
# cx1,cy1 = cMax[0][0]
# cx2,cy2 = cMax[-1][0]
# print(cx2, cy2) 
# print(cMax)
# print(x,y)
# for cnt in c:
# 	M = cv2.moments(cnt)
# 	if(M['m00']>800) and :
# 		cx = int(M['m10']/M['m00'])
# 		cy = int(M['m01']/M['m00'])
# 		cv2.drawContours(in_im, c, -1, (255, 0, 0), 2)


# cv2.imshow("Original",in_im)
# cv2.imshow("Binary",canny)
# # cv2.imshow("Eroded",eroded)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
