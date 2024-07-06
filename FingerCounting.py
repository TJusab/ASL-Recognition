import cv2
import time
import os
import HandTrackingModule as htm

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

# folder_path = "ASL_Images"
# my_list = os.listdir(folder_path)
# print(my_list)
# overlay_list = []
# for image_path in my_list:
#     image = cv2.imread(folder_path + "/" + image_path)
#     overlay_list.append(image)

previous_time = 0

detector = htm.HandDetector(detection_confidence=0.75)

fingertip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        fingers = []

        for id_number in range(1, 5):
            if landmark_list[fingertip_ids[id_number]][2] < landmark_list[fingertip_ids[id_number] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        print(fingers)


    # Display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
