import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import sys

brushThickness = 25
eraserThickness = 100


folderPath = "header"
myList = os.listdir(folderPath)
printimgCanvas = np.zeros((720, 1280, 3), np.uint8)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))
head = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(-1)
cap.set(3,1080)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85,maxHands=1)
xp, yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    success, img = cap.read()
    img = cv2.flip(img,1)


    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
       # print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        fingers = detector.fingerup()

        #print(fingers)
        

        if fingers[1] and fingers[2]:
           # xp, yp = 0, 0
            print("Selection mode")

            
            if y1 <121:
                if 250<x1<450:
                    head = overlayList[0]
                    drawColor = (255,0,255)

                elif 550 < x1 < 750:
                    head = overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 950:
                    head = overlayList[2]
                    drawColor = (0,255,0)
                elif 1050 < x1 < 1200:

                    head = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)


        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15,drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                  xp,yp = x1, y1
            cv2.line(img, (xp,yp), (x1,y1),drawColor,brushThickness)

            if drawColor == (0,0,0):
                  cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                  cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:    
                  cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                  cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)

            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
 

  

    img[0:121,0:1280] = head
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("camvas",imgCanvas)
    cv2.imshow("Inv",imgInv)
    cv2.waitKey(1)

