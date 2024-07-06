import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_confidence,
                                         min_tracking_confidence=self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id_number, landmark in enumerate(hand.landmark):
                height, width, channel = img.shape
                x_pos, y_pos = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id_number, x_pos, y_pos])
                if draw:
                    cv2.circle(img, (x_pos, y_pos), 15, (255, 0, 255), cv2.FILLED)

        return landmark_list


def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])

        # Setup FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display video footage
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
