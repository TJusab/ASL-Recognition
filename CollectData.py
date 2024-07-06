import cv2
import os

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
number_of_classes = 24
dataset_size = 100

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.mkdir(os.path.join(DATA_DIR, str(j)))

    for hand in ['right', 'left']:
        print(f'Collecting {hand} hand data for class {j}')

        while True:
            success, img = cap.read()
            cv2.imshow('Image', img)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            success, img = cap.read()
            if not success:
                continue
            cv2.imshow('Image', img)
            if cv2.waitKey(25) == ord('s'):
                filename = os.path.join(DATA_DIR, str(j), f'{hand}_{counter}.jpg')
                cv2.imwrite(filename, img)
                print(f'Picture {counter} of {hand} hand saved for class {j}')
                counter += 1
            if cv2.waitKey(25) == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()