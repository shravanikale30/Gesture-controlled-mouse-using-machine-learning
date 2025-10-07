import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KBController, Key
import pyautogui
import time
import os

# Initialize controllers
mouse = Controller()
keyboard = KBController()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Screen size
screen_w, screen_h = pyautogui.size()

# Smoothing
prev_x, prev_y = 0, 0
smoothening = 5

# Gesture flags
palm_triggered = False
fist_triggered = False
scroll_triggered = False
screenshot_triggered = False

# Last gesture text
gesture_text = "No Gesture Detected"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ---- Mouse Movement (Index Finger) ----
            index_x = int(hand_landmarks.landmark[8].x * screen_w)
            index_y = int(hand_landmarks.landmark[8].y * screen_h)
            curr_x = prev_x + (index_x - prev_x) / smoothening
            curr_y = prev_y + (index_y - prev_y) / smoothening
            mouse.position = (curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # ---- Finger Status ----
            index_ext  = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_ext = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_ext   = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            pinky_ext  = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            thumb_ext  = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x  # optional horizontal check

            # ---- Open Palm (All 5 Fingers) ----
            if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext and not palm_triggered:
                mouse.click(Button.left, 2)
                palm_triggered = True
                gesture_text = "Open Palm → Open File"
            elif not (thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext):
                palm_triggered = False

            # ---- Fist (No Fingers Extended) ----
            if not thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext and not fist_triggered:
                keyboard.press(Key.cmd)
                keyboard.press('w')
                keyboard.release('w')
                keyboard.release(Key.cmd)
                fist_triggered = True
                gesture_text = "Fist → Close Window"
            elif thumb_ext or index_ext or middle_ext or ring_ext or pinky_ext:
                fist_triggered = False

            # ---- Scroll Gestures ----
            # Scroll Down: Index + Middle + Ring
            if index_ext and middle_ext and ring_ext and not pinky_ext and not scroll_triggered:
                pyautogui.scroll(-300)
                gesture_text = "Scroll Down → Index+Middle+Ring"
                scroll_triggered = True
            # Scroll Up: Middle + Ring + Pinky
            elif not index_ext and middle_ext and ring_ext and pinky_ext and not scroll_triggered:
                pyautogui.scroll(300)
                gesture_text = "Scroll Up → Middle+Ring+Pinky"
                scroll_triggered = True
            elif not ((index_ext and middle_ext and ring_ext) or (not index_ext and middle_ext and ring_ext and pinky_ext)):
                scroll_triggered = False

            # ---- Screenshot Gesture: Index + Thumb ----
            if index_ext and thumb_ext and not middle_ext and not ring_ext and not pinky_ext and not screenshot_triggered:
                # Save directly to Desktop
                filename = f"screenshot_{int(time.time())}.png"
                save_path = os.path.join(os.path.expanduser("~/Desktop/lyproject/screenshots"), filename)
                pyautogui.screenshot(save_path)

                gesture_text = f"Screenshot Taken → {save_path}"
                screenshot_triggered = True
            elif not (index_ext and thumb_ext and not middle_ext and not ring_ext and not pinky_ext):
                screenshot_triggered = False

    # Display gesture text permanently until next gesture
    cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
