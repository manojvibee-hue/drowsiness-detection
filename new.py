import cv2
import mediapipe as mp
import serial
import time
import numpy as np

try:
    arduino = serial.Serial('COM5', 9600, timeout=1)  
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

mp_drawing = mp.solutions.drawing_utils

# EAR Function
def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    vertical1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    vertical2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    horizontal = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Driver Eye Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Driver Eye Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

CLOSED_EAR_THRESH = 0.21
CLOSED_DURATION = 3
ANGLE_THRESH = 45
ANGLE_DURATION = 3
NO_FACE_TIMEOUT = 5

eye_closed_time = None
angle_timer = None
last_face_time = time.time()
buzzer_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    trigger_buzzer = False

    if results.multi_face_landmarks:
        last_face_time = time.time()

        for face in results.multi_face_landmarks:
            lm = face.landmark

            # EAR Calculation
            leftEAR = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            rightEAR = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # Eye Color Logic
            if avgEAR < CLOSED_EAR_THRESH:
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                elapsed = time.time() - eye_closed_time

                if elapsed >= CLOSED_DURATION:
                    msg = "Driver Slept"
                    color = (0,0,255)
                    trigger_buzzer = True
                else:
                    msg = f"Eyes Closing: {elapsed:.1f}s"
                    color = (0,255,255)
            else:
                msg = "All Normal"
                color = (0,255,0)
                eye_closed_time = None

            cv2.putText(frame, msg, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

           
            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_LEFT_EYE,
                                      mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_RIGHT_EYE,
                                      mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1))

          
            left_ear = np.array([lm[234].x*w, lm[234].y*h])
            right_ear = np.array([lm[454].x*w, lm[454].y*h])
            neck = (left_ear + right_ear)/2 + np.array([0,50])
            left_shoulder = neck + np.array([-100,0])
            right_shoulder = neck + np.array([100,0])

            angle_left = angle_between(left_ear, neck, left_shoulder)
            angle_right = angle_between(right_ear, neck, right_shoulder)
            head_angle = max(angle_left, angle_right)

            cv2.putText(frame, f"Head Angle: {head_angle:.1f}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2)

            line_color = (0,255,0) if head_angle < ANGLE_THRESH else (0,0,255)

            cv2.line(frame, tuple(left_ear.astype(int)), tuple(neck.astype(int)), line_color, 3)
            cv2.line(frame, tuple(neck.astype(int)), tuple(left_shoulder.astype(int)), line_color, 3)
            cv2.line(frame, tuple(right_ear.astype(int)), tuple(neck.astype(int)), line_color, 3)
            cv2.line(frame, tuple(neck.astype(int)), tuple(right_shoulder.astype(int)), line_color, 3)

            if head_angle > ANGLE_THRESH:
                if angle_timer is None:
                    angle_timer = time.time()
                elapsed_angle = time.time() - angle_timer
                cv2.putText(frame, f"Head Down: {elapsed_angle:.1f}s", (50,190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
                if elapsed_angle >= ANGLE_DURATION:
                    trigger_buzzer = True
            else:
                angle_timer = None

    else:
        if time.time() - last_face_time > NO_FACE_TIMEOUT:
            cv2.putText(frame, "No Face Detected!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0,0,255),3)

   
    if trigger_buzzer:
        if not buzzer_on and arduino:
            arduino.write(b'1')
        buzzer_on = True
    else:
        if buzzer_on and arduino:
            arduino.write(b'0')
        buzzer_on = False

    cv2.imshow("Driver Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == 8:
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
