import cv2
import numpy as np
import os
import HandTrackingModule as htm

# Configuration
brushThickness = 25
eraserThickness = 100

# Load Header Images
folderPath = "Header"
overlayList = [cv2.imread(f'{folderPath}/{imgPath}') for imgPath in sorted(os.listdir(folderPath))]
header = overlayList[0]
drawColor = (255, 0, 255)

# Camera Setup
cap = cv2.VideoCapture(0)  # Use 1 if 0 doesn't work
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # Selection Mode: both index and middle fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)  # Pink
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)    # Blue
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)    # Green
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)      # Eraser (black)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode: only index finger up
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1

    # Combine image and canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Add header
    img[0:125, 0:1280] = header

    cv2.imshow("Painter", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
