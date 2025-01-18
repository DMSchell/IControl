import cv2
import pyautogui
import numpy as np
import dlib
import time
import threading

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

screen_width, screen_height = pyautogui.size()
calibrated = False

BLINK_THRESHOLD = 2
blink_frames = 0

pyautogui.FAILSAFE = False

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    finalEye = (A + B) / (2.0 * C)
    return finalEye

global mouse_x
global mouse_y
leftX = 1
rightX = 1
upY = 1
downY = 1

def mov_mouse(newX, newY, FPS):
    pyautogui.moveTo(newX, newY, duration=(1/20))

def start_mouse_move(newX, newY, FPS):
    mouse_thread = threading.Thread(target=mov_mouse, args=(newX, newY, FPS), daemon=True)
    mouse_thread.start()

def get_mouse_coords(eye_center, frame_width, frame_height):
    eye_x, eye_y = eye_center
    return int(screen_width * (eye_x / frame_width)), int(screen_height * (eye_y / frame_height))

def map_gaze_to_mouse(mouse_x, mouse_y):
    if calibrated:
        MXpos = ((mouse_x - leftX)*screen_width)/(rightX - leftX)
        MYpos = ((mouse_y - upY)*screen_height)/(downY - upY)
        start_mouse_move(MXpos, MYpos, fps)
    else:
        start_mouse_move(screen_width - mouse_x, mouse_y, fps)
    
start_time = time.time()
frame_count = 0
fps = 1
previousface = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray)

    frame_count += 1

    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0
    cv2.putText(frame, "fps:  " + str(fps), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    for face in faces:
        if face == faces[0]:
            landmarks = predictor(frame_gray, face)
            for l in range(0, 67):
                if previousface != []:
                    smoothing = 0.3
                    smoothed_point = smoothing * landmarks.part[l] + (1 - smoothing) * previousface[l]
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < 0.2:
                blink_frames += 1
            else:
                if blink_frames >= BLINK_THRESHOLD and calibrated:
                    pyautogui.click()
                blink_frames = 0

            left_eye_center = np.mean(left_eye, axis=0).astype(float)
            right_eye_center = np.mean(right_eye, axis=0).astype(float)

            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            #cv2.circle(frame, tuple(left_eye_center), 3, (255, 0, 0), -1)
            #cv2.circle(frame, tuple(right_eye_center), 3, (255, 0, 0), -1)

            combined_eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)
            mouse_x, mouse_y = get_mouse_coords(combined_eye_center, frame.shape[1], frame.shape[0])
            map_gaze_to_mouse(mouse_x, mouse_y)
    cv2.imshow("tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if leftX == 1:
            leftX = mouse_x
        elif rightX == 1:
            rightX = mouse_x
        elif upY == 1:
            upY = mouse_y
        elif downY == 1:
            downY = mouse_y
            calibrated = True

cap.release()
cv2.destroyAllWindows()
