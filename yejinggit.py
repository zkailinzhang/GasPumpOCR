from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import math



#step1 颜色分割， 扩大一圈
path = "./yejingtestimg/ya0106.png"
#path = "./yejing03.png"
img_raw = cv2.imread(path)


ocr = PaddleOCR()
print(ocr.ocr(path))
result = ocr.ocr(path,rec=False)
image = Image.open(path).convert('RGB')
im_show = draw_ocr(image,result,txts=None,scores=None,font_path="/home/zkl/code/PaddleOCR-release-2.1/doc/fonts/simfang.ttf")
im_show2 = Image.fromarray(im_show)
im_show2.show()



lower_red = np.array([2,55,100])
upper_red = np.array([7,90,120])
HSV = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
img2 = cv2.inRange(HSV, lower_red, upper_red) 

kernel = np.ones((7,7),np.uint8)
 
dst = cv2.dilate(img2,kernel)


contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

yejingx,yejingy,dotw,doth =0,0,0,0
yejing_only =None


#找yejing
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1800:  
        [x, y, w, h] = cv2.boundingRect(contour)
        img_roi = img_raw[y-20:y + h+20,x-25:x + w+25]  
        yejingx,yejingy,dotw,doth =x,y, w-10, h-10
    
        cv2.rectangle(img_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)            
        yejing_only =img_raw[y+5:y + h-5,x+5:x + w-5]

ocr = PaddleOCR()
rst = ocr.ocr(img_roi)

digits =rst[0][1][0]
digitnum = len(digits)
startx = rst[0][0][0][0]
lengthx = rst[0][0][1][0] - rst[0][0][0][0]
starty = rst[0][0][0][1]
lengthy = rst[0][0][3][1] - rst[0][0][0][1]

fenwei = [startx + lengthx/3,startx + 2*lengthx/3]



alpha = float(2.5)

exposure_img = cv2.multiply(yejing_only, np.array([alpha]))

img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)

blur = 5  
img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)

cropped = img_blurred


threshold=15  
adjustment=6  
cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            threshold, adjustment)
erode=3 
iterations = 4 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)

inverse =   (255 - eroded) 


contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


dotposition=0

#找小数点
for c in contours:
    area = cv2.contourArea(c)
    [x, y, w, h] = cv2.boundingRect(c)
    if area < 500 and y> (0+doth/2):  #2000  框出小数点，0和7  没有8    500，就剩小数点一个了 #第二个过滤条件，横坐标 在下方，
        cv2.drawContours(yejing_only, [c], -1, (36,255,12), -1)
        dotposition = np.argmin(np.array(fenwei)-20-x)
        #cv2.imwrite('dot_only3.jpg', yejing_only)

result_number = digits[dotposition] +'.'+digits[dotposition+1:]
print(result_number)
cv2.putText(img_raw, '%s' %(result_number), (yejingx,yejingy), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)    


cv2.imshow('yejing03', img_raw)
cv2.waitKey(0)
cv2.imwrite('ya0106-dushu.jpg', img_raw)

#result:  0.78