import cv2
import time
import handTrackingModule as ht

cap = cv2.VideoCapture(1)
pTime = 0
cTime = 0
detector = ht.handDtetcor()
while True:
    print()
    ret, image = cap.read()
    if not ret:
        break
    image = detector.findHands(image)
    landmarks = detector.findLandmarks(image)
    if len(landmarks) != 0:
        print(landmarks[0])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )
    # Display the image
    cv2.imshow("image", image)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
