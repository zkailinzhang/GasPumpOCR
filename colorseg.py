from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import math



path = "./testimg.png"
 
# 读取图片并缩放方便显示
img = cv2.imread(path)

#bgr  85-100，85-110，120-130

#ret,img2 = cv2.threshold(img,(85,85,120), (100,110,130), cv2.THRESH_BINARY)
lower_red = np.array([2,55,100])
upper_red = np.array([7,90,120])
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2 = cv2.inRange(HSV, lower_red, upper_red) 

kernel = np.ones((7,7),np.uint8)
 
# ## c.图像的腐蚀，默认迭代次数
# erosion = cv2.erode(src,kernel)
 
## 图像的膨胀
dst = cv2.dilate(img2,kernel)
cv2.imshow('image', dst)
cv2.waitKey(0)

contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


for contour in contours:
    # get rectangle bounding contour
    #[x, y, w, h] = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    [x, y, w, h] = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('image1', img)
    cv2.waitKey(0) 


for contour in contours:
    # get rectangle bounding contour
    #[x, y, w, h] = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    #print(x, y, w, h)
    if area > 2000:  #2000  框出小数点，0和7  没有8    500，就剩小数点一个了
        [x, y, w, h] = cv2.boundingRect(contour)
        #cv2.drawContours(img, [contour], -1, (36,255,12), -1)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        img_roi        = img[y-10:y + h+10,x-10:x + w+10]
       # cv2.imshow('image1', img_roi)
        #cv2.waitKey(0) 
        cv2.imwrite("yejing_only.jpg",img_roi)
   



# lower_red = np.array([70,70,90])
# upper_red = np.array([100,105,115])
# img2 = cv2.inRange(img, lower_red, upper_red) 

# cv2.imshow('image', img2)
# cv2.waitKey(0)




height, width = img.shape[:2]
# size = (int(width * 0.2), int(height * 0.2))
# # 缩放
# img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
 
# BGR转化为HSV
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
# 鼠标点击响应事件
def getposHsv(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("HSV is", HSV[y, x])
 
 
def getposBgr(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("Bgr is", img[y, x])
 
 
cv2.imshow("imageHSV", HSV)
cv2.imshow('image', img)
cv2.setMouseCallback("imageHSV", getposHsv)
cv2.setMouseCallback("image", getposBgr)
cv2.waitKey(0)