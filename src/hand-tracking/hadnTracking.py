import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

green_color = (0, 255, 0)
red_color = (0, 0, 255)
landmark_drawing_spec = mpDraw.DrawingSpec(
    color=red_color, thickness=2, circle_radius=2
)
connection_drawing_spec = mpDraw.DrawingSpec(color=green_color, thickness=2)

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Convert image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(img_rgb)

    # Draw the landmarks and connections
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = image.shape
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                # print(id, center_x, center_y)
                # if id == 4:
                # cv2.circle(image, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(
                image,
                handLandmarks,
                mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec,
                connection_drawing_spec,
            )
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    print("fps", fps)
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
