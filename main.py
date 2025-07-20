import cv2
import mediapipe as mp
import time
from directkeys import RIGHT_ARROW, LEFT_ARROW, PressKey, ReleaseKey

# Key mappings
brake_key = LEFT_ARROW
gas_key = RIGHT_ARROW

# Initial delay to give time to switch to the game window
time.sleep(2.0)

# Track currently pressed keys
current_keys_pressed = set()

# MediaPipe setup
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tip_ids = [4, 8, 12, 16, 20]

# Start webcam
video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        if not ret:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)

        fingers = []
        if lm_list:
            # Thumb
            fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
            # Other 4 fingers
            for i in range(1, 5):
                fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2] else 0)

        total_fingers = fingers.count(1)

        key_pressed_now = None

        if total_fingers == 0:
            # Brake
            cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            PressKey(brake_key)
            key_pressed_now = brake_key

        elif total_fingers == 5:
            # Gas
            cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, "  GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            PressKey(gas_key)
            key_pressed_now = gas_key

        # Handle key release
        if key_pressed_now:
            if key_pressed_now not in current_keys_pressed:
                # Release any previously pressed key
                for key in current_keys_pressed:
                    if key != key_pressed_now:
                        ReleaseKey(key)
                current_keys_pressed = {key_pressed_now}
        else:
            # No valid gesture → release all
            for key in current_keys_pressed:
                ReleaseKey(key)
            current_keys_pressed.clear()

        # Show video
        cv2.imshow("Hand Gesture Controller", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
video.release()
cv2.destroyAllWindows()
for key in current_keys_pressed:
    ReleaseKey(key)
