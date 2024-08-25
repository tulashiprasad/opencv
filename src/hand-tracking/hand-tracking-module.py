import cv2
import mediapipe as mp
import time


class handDtetcor:
    def __init__(
        self,
        staticImageMode=False,
        maxHands=2,
        modelComplexicity=1,
        minDetectionConfidence=0.5,
        minTrackingConfidence=0.5,
    ):
        self.mode = staticImageMode
        self.maxHnads = maxHands
        self.modelComplexity = modelComplexicity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        # Initialize MediaPipe hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHnads,
            self.modelComplexity,
            self.minDetectionConfidence,
            self.minTrackingConfidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):
        green_color = (0, 255, 0)
        red_color = (0, 0, 255)
        landmark_drawing_spec = self.mpDraw.DrawingSpec(
            color=red_color, thickness=2, circle_radius=2
        )
        connection_drawing_spec = self.mpDraw.DrawingSpec(
            color=green_color, thickness=2
        )

        # Convert image to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        self.results = self.hands.process(img_rgb)

        # Draw the landmarks and connections
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image,
                        handLandmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec,
                        connection_drawing_spec,
                    )
        return image

    def findLandmarks(self, image, handNo=0, landMarkn0=0, draw=True):
        # for hand landmarks
        landmarks = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                # print(id, center_x, center_y)
                landmarks.append([id, center_x, center_y])
                # if id == 4:
                # if draw and id == landMarkn0:
                if draw:
                    cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), cv2.FILLED)
        return landmarks


def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0
    detector = handDtetcor()
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = detector.findHands(image)
        landmarks = detector.findLandmarks(image)
        print(landmarks)
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


if __name__ == "__main__":
    main()
