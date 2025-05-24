import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def get_distance(lm1, lm2):
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5

light_state = "OFF"
fan_state = "OFF"
gesture_active = {"light": False, "fan": False}

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]
            ring_tip = handLms.landmark[16]

            dist_thumb_index = get_distance(thumb_tip, index_tip)
            dist_thumb_ring = get_distance(thumb_tip, ring_tip)

            # Light Gesture: Thumb + Index
            if dist_thumb_index < 0.05:
                if not gesture_active["light"]:
                    light_state = "OFF" if light_state == "ON" else "ON"
                    print(f"Light: TURNED {light_state}")
                    gesture_active["light"] = True
            else:
                gesture_active["light"] = False

            # Fan Gesture: Thumb + Ring
            if dist_thumb_ring < 0.05:
                if not gesture_active["fan"]:
                    fan_state = "OFF" if fan_state == "ON" else "ON"
                    print(f"Fan: TURNED {fan_state}")
                    gesture_active["fan"] = True
            else:
                gesture_active["fan"] = False

    cv2.imshow("Hand Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()