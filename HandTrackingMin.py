import cv2
import mediapipe as mp
import time

# Get camera capture from webcam
cap = cv2.VideoCapture(0)

# Get hands model from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    # Read the image
    success, img = cap.read()

    # Convert the image to RGB for hands object
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame and get results
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandmarks.landmark):
                height, width, channel = img.shape
                centerX, centerY = int(landmark.x*width), int(landmark.y*height)
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    # Setup FPS
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Display video footage
    cv2.imshow('Image', img)
    cv2.waitKey(1)
