import cv2
import numpy as np



coins_img_gray = cv2.imread('lcdrst.jpg', 0)
coins_img = cv2.imread('lcdrst.jpg')

coins_circle = cv2.HoughCircles(coins_img_gray, 
                                cv2.HOUGH_GRADIENT, 
                                1, 
                                10, 
                                param1=50, 
                                param2=100, 
                                minRadius=0, 
                                maxRadius=500)

circles = coins_circle.reshape(-1, 3)
circles = np.uint16(np.around(circles))

for i in circles:
    cv2.circle(coins_img, (i[0], i[1]), i[2], (0, 0, 255), 5)   # 画圆
    cv2.circle(coins_img, (i[0], i[1]), 2, (0, 255, 0), 10)     # 画圆心

cv2.imshow('', coins_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
