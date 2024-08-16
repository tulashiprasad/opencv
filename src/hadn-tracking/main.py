import cv2
import mediapipe as mp

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Initialize MediaPipe drawing utilities
mpDraw = mp.solutions.drawing_utils

# Create a DrawingSpec for green color
green_color = (0, 255, 0)  # Green in BGR format
landmark_drawing_spec = mpDraw.DrawingSpec(color=green_color, thickness=2, circle_radius=2)
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
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                image, 
                landmarks, 
                mpHands.HAND_CONNECTIONS, 
                landmark_drawing_spec, 
                connection_drawing_spec
            )

    # Display the image
    cv2.imshow("image", image)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()