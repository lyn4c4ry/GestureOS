import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Camera setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Transparent drawing layer overlaid on the camera feed
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# ── Constants ──────────────────────────────────────────────────────────────────
FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

DRAW_COLOR     = (0, 255, 0)
DRAW_THICKNESS = 4
ERASER_SIZE    = 40

# ── State ──────────────────────────────────────────────────────────────────────
mode       = "FREE"  # DRAW | FREE | ERASE
prev_point = None    # last drawn point for continuous lines


# ── Helper functions ───────────────────────────────────────────────────────────

def get_pos(landmark, w, h):
    """Convert a normalized landmark to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)


def only_index_up(landmarks):
    """Return True if only the index finger is extended."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y > landmarks[10].y,
        landmarks[16].y > landmarks[14].y,
        landmarks[20].y > landmarks[18].y,
    ])


def is_open_hand(landmarks):
    """Return True if all four fingers are extended (open palm)."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y,
    ])


def hand_center(landmarks, w, h):
    """Return the center pixel of the palm."""
    points = [0, 5, 9, 13, 17]
    cx = int(sum(landmarks[i].x for i in points) / len(points) * w)
    cy = int(sum(landmarks[i].y for i in points) / len(points) * h)
    return cx, cy


def hand_size(landmarks, w, h):
    """Estimate palm width in pixels (used for eraser radius)."""
    x1, y1 = int(landmarks[5].x * w),  int(landmarks[5].y * h)
    x2, y2 = int(landmarks[17].x * w), int(landmarks[17].y * h)
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def draw_hand_labels(frame, hand_landmarks, label, w, h):
    """Render hand label and fingertip names on the frame."""
    color = (255, 150, 0) if label == "LEFT" else (0, 150, 255)
    wx = int(hand_landmarks.landmark[0].x * w)
    wy = int(hand_landmarks.landmark[0].y * h)
    cv2.putText(frame, label, (wx - 20, wy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for idx, name in FINGER_NAMES.items():
        x = int(hand_landmarks.landmark[idx].x * w)
        y = int(hand_landmarks.landmark[idx].y * h)
        cv2.putText(frame, name, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_landmarks  = None
    right_landmarks = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label_raw = results.multi_handedness[i].classification[0].label
            label     = "LEFT" if label_raw == "Left" else "RIGHT"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw_hand_labels(frame, hand_landmarks, label, w, h)

            if label == "LEFT":
                left_landmarks = hand_landmarks.landmark

            elif label == "RIGHT":
                # Right hand gesture controls the active drawing mode:
                #   1 finger  -> DRAW
                #   2 fingers -> FREE (pen up)
                #   3 fingers -> ERASE
                right_landmarks = hand_landmarks.landmark
                lm     = hand_landmarks.landmark
                index  = lm[8].y  < lm[6].y
                middle = lm[12].y < lm[10].y
                ring   = lm[16].y < lm[14].y
                pinky  = lm[20].y < lm[18].y

                if index and not middle and not ring and not pinky:
                    mode = "DRAW"
                elif index and middle and not ring and not pinky:
                    mode = "FREE"
                elif index and middle and ring and not pinky:
                    mode = "ERASE"

    # ── Drawing logic (left hand) ──────────────────────────────────────────────
    if left_landmarks:
        if mode == "DRAW":
            if only_index_up(left_landmarks):
                draw_point = get_pos(left_landmarks[8], w, h)
                if prev_point:
                    cv2.line(canvas, prev_point, draw_point, DRAW_COLOR, DRAW_THICKNESS)
                prev_point = draw_point
                cv2.circle(frame, draw_point, 8, DRAW_COLOR, -1)
            else:
                prev_point = None

        elif mode == "FREE":
            prev_point = None

        elif mode == "ERASE":
            if is_open_hand(left_landmarks):
                # Open palm -> large circular eraser
                size   = hand_size(left_landmarks, w, h)
                center = hand_center(left_landmarks, w, h)
                cv2.circle(canvas, center, size, (0, 0, 0), -1)
                cv2.circle(frame,  center, size, (0, 0, 255), 2)
                prev_point = None
            elif only_index_up(left_landmarks):
                # Index finger only -> precise eraser stroke
                draw_point = get_pos(left_landmarks[8], w, h)
                if prev_point:
                    cv2.line(canvas, prev_point, draw_point, (0, 0, 0), ERASER_SIZE)
                prev_point = draw_point
                cv2.circle(frame, draw_point, ERASER_SIZE // 2, (0, 0, 255), 2)
            else:
                prev_point = None
    else:
        prev_point = None

    # ── Merge canvas onto frame ────────────────────────────────────────────────
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas[mask > 0]

    # ── HUD ───────────────────────────────────────────────────────────────────
    mode_colors = {"DRAW": (0, 255, 0), "FREE": (200, 200, 200), "ERASE": (0, 0, 255)}
    cv2.rectangle(frame, (10, 10), (300, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"MODE: {mode}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_colors[mode], 2)
    cv2.putText(frame, "RIGHT: 1=Draw  2=Free  3=Erase  |  C: Clear  |  Q: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imshow("Hand Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()