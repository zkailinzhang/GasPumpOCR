import cv2
import numpy as np

image = cv2.imread('lcdrst.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.imshow('opening1', opening)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    area = cv2.contourArea(c)
    print(area)
    if area < 2000:
        cv2.drawContours(original, [c], -1, (36,255,12), -1)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('original', original)
cv2.waitKey()