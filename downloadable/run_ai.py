import os
os.environ["QT_QPA_PLATFORM"] = "xcb" # Linux Wayland/Qt fix

import cv2
import mediapipe as mp
import pickle
import pyautogui
import time

# --- SETUP MOUSE CONTROLLER ---
# 1. FAILSAFE: Move mouse to any corner of screen quickly to crash script.
pyautogui.FAILSAFE = True 
# Get screen resolution
screen_w, screen_h = pyautogui.size()
# Cooldown for clicking so we don't click too fast (in seconds)
CLICK_COOLDOWN = 1.0 
last_click_time = 0

# --- LOAD AI BRAIN ---
print("Loading AI Brain...")
try:
    with open('hand_model_3d.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Brain loaded successfully!")
except FileNotFoundError:
    print("Error: 'hand_model_3d.pkl' not found. Train the model first!")
    exit()

# --- SETUP MEDIAPIPE ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

print("Starting combined Visual Tracker & Mouse Control...")
print("- Move mouse by pointing your Index finger.")
print("- Left Click by making a Fist (detected as '0' fingers).")
print("- Press 'q' in the window or Ctrl+C in terminal to stop.")

# We increase confidence slightly for mouse control stability
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Mirror effect and colors
        image = cv2.flip(image, 1)
        # Get frame dimensions for internal coordinate mapping
        frame_h, frame_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- VISUALS (You want to keep these!) ---
                # Draw the skeleton on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- 1. MOUSE MOVEMENT LOGIC ---
                # Grab the Tip of the Index Finger (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                
                # Convert MediaPipe percentages (0.0 to 1.0) to actual Screen Pixel coordinates
                mouse_x = int(index_finger_tip.x * screen_w)
                mouse_y = int(index_finger_tip.y * screen_h)
                
                # Move the OS mouse instantly to that pixel
                # _pause=False ensures maximum speed without unnecessary delay
                pyautogui.moveTo(mouse_x, mouse_y, _pause=False)

                # --- 2. AI GESTURE RECOGNITION (WRIST RELATIVITY) ---
                # This is essential for both your visual text and your clicks!
                landmark_data = []
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

                for landmark in hand_landmarks.landmark:
                    rel_x = landmark.x - wrist_x
                    rel_y = landmark.y - wrist_y
                    rel_z = landmark.z - wrist_z
                    landmark_data.extend([rel_x, rel_y, rel_z])
                
                # Ask the brain how many fingers are up (must pass a 2D array)
                prediction = model.predict([landmark_data])[0]

                # --- 3. MOUSE CLICKING LOGIC ---
                color = (0, 255, 0) # Normal green color
                text_prefix = "Fingers:"
                
                current_time = time.time()
                # If AI detects a fist ('0') AND the cooldown period has passed
                if prediction == '0':
                    if (current_time - last_click_time) > CLICK_COOLDOWN:
                        print("AI detected Fist: --> LEFT CLICK! <--")
                        pyautogui.click()
                        last_click_time = current_time # Reset timer
                    
                    # Update visuals briefly during a click detection
                    color = (0, 0, 255) # Red for active click detection
                    text_prefix = "CLICK! -> "
                
                # --- VISUALS (Text Overlay for Testing) ---
                cv2.putText(image, f'{text_prefix} {prediction}', (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                
                # Highlight index fingertip being used for mouse control
                # Draw a small purple circle over it for visual testing
                ix = int(index_finger_tip.x * frame_w)
                iy = int(index_finger_tip.y * frame_h)
                cv2.circle(image, (ix, iy), 10, (255, 0, 255), -1)
                
                break # Just track one hand for now

        # Show the visual output window
        cv2.imshow('Testing Visuals & Mouse Control', image)

        # Handle keyboard input within OpenCV window
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()