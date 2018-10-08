import cv2

def create_row(img,lines,row_size):
    row,col,_ = img.shape
    initial_y = lines[0][0][1]

    while initial_y < row:

        roi = img[initial_y:initial_y+row_size,:,:]
        initial_y += row_size
        roi_row,roi_col,_ = roi.shape
        #print(row)
        if roi_row > 10:
            cv2.imshow("ROI",roi)
            print(roi.shape)
            cv2.waitKey(0)
