import cv2
from PIL import Image
import numpy as np


yellow = [0, 255, 0]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 160, 20])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsvImage, lower_red, upper_red)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        print(bbox)
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("frame", result)
        cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
