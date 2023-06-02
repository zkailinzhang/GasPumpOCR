from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import math


#step1 颜色分割， 扩大一圈
path = "./testimg.png"
img = cv2.imread(path)

lower_red = np.array([2,55,100])
upper_red = np.array([7,90,120])
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2 = cv2.inRange(HSV, lower_red, upper_red) 

kernel = np.ones((7,7),np.uint8)
 
dst = cv2.dilate(img2,kernel)


contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

dotx,doty,dotw,doth =0,0,0,0

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 2000:  #2000  框出小数点，0和7  没有8    500，就剩小数点一个了
        [x, y, w, h] = cv2.boundingRect(contour)
        
        #img_roi = img[y-10:y + h+10,x-10:x + w+10]  # 直接喂入 paddle 178  
        img_roi = img[y-20:y + h+20,x-20:x + w+20]  # 直接喂入 paddle 078     原图也是078    也就是说 还是大点好图片
        dotx,doty,dotw,doth =0,0, w-10, h-10
        cv2.imwrite("yejing_only.jpg",img[y+5:y + h-5,x+5:x + w-5])


ocr = PaddleOCR()
rst = ocr.ocr(img_roi)

digits =rst[0][1][0]
digitnum = len(digits)
startx = rst[0][0][0][0]
lengthx = rst[0][0][1][0] - rst[0][0][0][0]
starty = rst[0][0][0][1]
lengthy = rst[0][0][3][1] - rst[0][0][0][1]

fenwei = [startx + lengthx/3,startx + 2*lengthx/3]
#仅仅数字的roi  rst[0][0][0]



#rst = ocr.ocr(path) 
#[[[[148.0, 32.0], [355.0, 31.0], [355.0, 116.0], [149.0, 117.0]], ('078', 0.814167)]]   顺时针
print(rst)
print(fenwei)














coins_img = cv2.imread('yejing_only.jpg')

img =coins_img.copy()



alpha = float(2.5)


# Adjust the exposure
exposure_img = cv2.multiply(img, np.array([alpha]))



# Convert to grayscale
img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)


# Blur to reduce noise
blur = 5  #0-25
img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)


cropped = img_blurred

# Threshold the image
threshold=15  #0-500
adjustment=6   #0-200
cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            threshold, adjustment)


# Erode the lcd digits to make them continuous for easier contouring
erode=3 #0-5
iterations = 4 # 0-5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)


# Reverse the image to so the white text is found when looking for the contours
inverse =   (255 - eroded) 

# Find the lcd digit contours
contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


dotposition=0

for c in contours:
    area = cv2.contourArea(c)
    print(area)
    [x, y, w, h] = cv2.boundingRect(c)
    if area < 500 and y> (doty+doth/2):  #2000  框出小数点，0和7  没有8    500，就剩小数点一个了
        cv2.drawContours(img, [c], -1, (36,255,12), -1)
        print("x",x)
        dotposition = np.argmin(np.array(fenwei)-20-x)
    #第二个过滤条件，横坐标 在下方，
'''  
64.0
1629.5
2510.5
64.0
918.5

[217.0, 286.0]
64.0
x 182
'''
cv2.imwrite('dot_only3.jpg', img)
cv2.imshow('x', img)
cv2.waitKey(0)

result_number = digits[dotposition] +'.'+digits[dotposition+1:]
print("result: ",result_number)
#result:  0.78