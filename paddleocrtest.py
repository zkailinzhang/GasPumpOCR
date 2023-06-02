from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy

'''
path = "./testimg.png"
img = cv2.imread(path)


ocr = PaddleOCR()

result = ocr.ocr(path,rec=False)



image = Image.open(path).convert('RGB')
im_show = draw_ocr(image,result,txts=None,scores=None,font_path="~/code/PaddleOCR-release-2.1/doc/fonts/simfang.ttf")


im_show2 = Image.fromarray(im_show)
#im_show2.show()

im_show2.save('LCD.jpg')


ocr.ocr(path)
rst = ocr.ocr(path)


In [20]: rst[:][4]
Out[20]: 
[[[204.0, 133.0], [383.0, 131.0], [384.0, 201.0], [205.0, 202.0]],
 ('078', 0.84615517)]

In [21]: rst[:][4][1]
Out[21]: ('078', 0.84615517)

In [22]: rst[:][4][1][0]
Out[22]: '078'


print(rst[:][4][1])
#('078', 0.84615517)

print(rst[:][4][1][0])
#'078'
cv2.imshow("x",img)
#cv2.waitKey()

lcd = img[131:202,204:385]
lcd = img[120:210,190:395]
# cv2.imshow("x",lcd)
# cv2.waitKey()
cv2.imwrite("./lcdrst.jpg",lcd)
'''


lcd = cv2.imread("./lcdrst.jpg")
gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(blurred, 120, 255, 1)
cv2.imshow("1", edged)
cv2.waitKey()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None


for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        displayCnt = approx
        break

warped = four_point_transform(gray, displayCnt.reshape(4, 2))

thresh = cv2.threshold(warped, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("2", thresh)

digit_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digit_cnts = imutils.grab_contours(digit_cnts)

threshold_max_area = 25
threshold_min_area = 5
contour_image = thresh.copy()

for c in digit_cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    area = cv2.contourArea(c) 
    if area < threshold_max_area and area > threshold_min_area:
        cv2.drawContours(contour_image,[c], 0, (100,5,10), 3)

cv2.imshow("detect decimal", contour_image)
cv2.waitKey(0)


'''
In [14]: ocr.ocr(path)
[2021/08/02 15:22:15] root INFO: dt_boxes num : 19, elapse : 0.2132253646850586
[2021/08/02 15:22:15] root INFO: rec_res num  : 19, elapse : 0.12542390823364258
Out[14]: 
[[[[137.0, 63.0], [350.0, 66.0], [350.0, 86.0], [137.0, 83.0]],
  ('DAM05数字电流表', 0.978155)],
 [[[390.0, 55.0], [442.0, 51.0], [444.0, 76.0], [392.0, 80.0]],
  ('MO', 0.66977894)],
 [[[39.0, 88.0], [91.0, 88.0], [91.0, 104.0], [39.0, 104.0]],
  ('双宝电子', 0.9888069)],
 [[[382.0, 86.0], [454.0, 86.0], [454.0, 104.0], [382.0, 104.0]],
  ('00000066', 0.9326773)],
 [[[204.0, 133.0], [383.0, 131.0], [384.0, 201.0], [205.0, 202.0]],
  ('078', 0.84615517)],
 [[[418.0, 142.0], [449.0, 143.0], [447.0, 195.0], [416.0, 194.0]],
  ('A', 0.9851423)],
 [[[54.0, 295.0], [73.0, 296.0], [69.0, 373.0], [50.0, 372.0]],
  ('效期7022', 0.7032961)],
 [[[80.0, 298.0], [93.0, 298.0], [93.0, 321.0], [80.0, 321.0]],
  ('0：', 0.80142915)],
 [[[171.0, 339.0], [318.0, 339.0], [318.0, 366.0], [171.0, 366.0]],
  ('整流器交流电流', 0.9972207)],
 [[[28.0, 349.0], [39.0, 349.0], [39.0, 358.0], [28.0, 358.0]],
  ('17', 0.5927587)],
 [[[71.0, 384.0], [105.0, 384.0], [105.0, 435.0], [71.0, 435.0]],
  ('15', 0.8597212)],
 [[[80.0, 424.0], [95.0, 424.0], [95.0, 439.0], [80.0, 439.0]],
  ('S', 0.778354)],
 [[[143.0, 415.0], [388.0, 414.0], [388.0, 432.0], [143.0, 433.0]],
  ('科技实业有限公司', 0.9994484)]]

'''