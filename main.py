import cv2
from tracker import *
#Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# Object detection from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)


    #Extract region of interest
    ROI = frame[340: 720, 500: 800]

    mask = object_detector.apply(ROI)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(ROI, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

            #print(x, y, w, h)


    #Object Tracking
    boxes_ids = tracker.update(detections)
    for boxes_id in boxes_ids:
        x, y ,w, h, id = boxes_id
        cv2.putText(ROI, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(ROI, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #print(boxes_ids)

    print(detections)
    cv2.imshow("ROI", ROI)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyWindow()
