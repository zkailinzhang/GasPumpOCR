from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import math

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

#step1 颜色分割， 扩大一圈
path = "./yejingtestimg/ya0108.png"
pathsave = './yejingtestimg/ya0108-dushu.jpg'
#path = "./yejing03.png"
img_raw = cv2.imread(path)


#for test
ocr = PaddleOCR()

# result = ocr.ocr(path,rec=False)
# image = Image.open(path).convert('RGB')
# im_show = draw_ocr(image,result,txts=None,scores=None,font_path="/home/zkl/code/PaddleOCR-release-2.1/doc/fonts/simfang.ttf")
# im_show2 = Image.fromarray(im_show)
# im_show2.show()

result  = ocr.ocr(path)
#print(result)

digitmaxlength = 5
digits= None
lablexy = None 

result  =np.array(result)
for ii in range(len(result)):
    tt = result[:,1][ii][0]
    #长度小于5个，是否数字，坐标上半部门，面积较大，长宽比阈值，是长方形
    if is_number(tt) and len(tt) < digitmaxlength:
        digits = tt
        lablexy = result[:,0][ii]
if len(digits) ==0: print("检测不到")
digitlens = len(digits)

startx = int(lablexy[0][0])
lengthx = int(lablexy[1][0] - lablexy[0][0])
starty = int(lablexy[0][1])
lengthy = int(lablexy[3][1] - lablexy[0][1])

fenwei = [ i*lengthx/digitlens for i in range(1,digitlens)]

x,y,w,h=startx,starty,lengthx,lengthy



yejing_only = img_raw[y-15:y + h+15,x-15:x + w+15]


alpha = float(2.5)
# Adjust the exposure
exposure_img = cv2.multiply(yejing_only, np.array([alpha]))
# Convert to grayscale
img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)
# Blr to reduce noise
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


dotposition = 0

#找小数点
for c in contours:
    area = cv2.contourArea(c)
    [x, y, w, h] = cv2.boundingRect(c)
    print("area: ",area)
    #2000  框出小数点，0和7  没有8    500，就剩小数点一个了 #第二个过滤条件，横坐标 在下方，
    if area < 500 and y> (0+lengthy/2):  
        
        dotposition = np.argmin(np.abs(np.array(fenwei)-15-x))
        
        # cv2.drawContours(yejing_only, [c], -1, (36,255,12), -1)
        # # #cv2.imwrite('dot_only3.jpg', yejing_only)
        # cv2.imshow('yejing03', yejing_only)
        # cv2.waitKey(0)

        result_number = digits[:dotposition+1] +'.'+digits[dotposition+1:]
        break
    else:
        result_number = digits

print("result_number: ",result_number)
#cv2.rectangle(img_raw, (x, y), (x + w, y + h), (0, 0, 255), 2) 
cv2.putText(img_raw, '%s' %(result_number), (startx,starty), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)    


cv2.imshow('yejing03', img_raw)
cv2.waitKey(10)
cv2.imwrite(pathsave, img_raw)

#result:  0.78