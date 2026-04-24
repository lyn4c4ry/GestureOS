import cv2
import mediapipe as mp
import numpy as np
import math
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, Tk

# ── Stdout encoding fix (Handles Windows-specific encoding issues) ─────────────
sys.stdout.reconfigure(encoding='utf-8')

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
# Set your desired resolution
WIDTH, HEIGHT = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Transparent drawing layer overlaid on the camera feed
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# ── Constants ──────────────────────────────────────────────────────────────────
FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
DRAW_COLOR     = (0, 255, 0)
DRAW_THICKNESS = 4
ERASER_SIZE    = 40

# Settings button region (top-right corner)
BTN_X1, BTN_Y1 = WIDTH - 150, 10
BTN_X2, BTN_Y2 = WIDTH - 10, 50

# ── Mutable app state ──────────────────────────────────────────────────────────
state = {
    "mode":            "FREE",   # DRAW | FREE | ERASE | INTERACTIVE
    "prev_point":      None,     # last drawn point for continuous lines
    "save_flash":      0,        # frame counter for the save confirmation flash
    "screenshots_dir": str(Path.home() / "Desktop"),
    "btn_clicked":     False,
    "mouse_pos":       (0, 0),
    "show_ui":         False,    # Feature to toggle settings overlay
    "fullscreen_toggle": True,   # Toggle fullscreen mode in settings
    "mode_menu_open":  False,    # Mode selection menu
    "ok_gesture_active": False,  # Track if OK gesture was active to prevent toggle spam
    "right_hand_mode": "FREE"    # Right hand mode (used for mode menu)
}
Path(state["screenshots_dir"]).mkdir(parents=True, exist_ok=True)

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
    """Render hand label and fingertip names on the frame with enhanced styling."""
    color = (255, 150, 0) if label == "LEFT" else (0, 150, 255)
    shadow_color = (50, 50, 50)
    wx = int(hand_landmarks.landmark[0].x * w)
    wy = int(hand_landmarks.landmark[0].y * h)
    # Shadow effect for hand label
    cv2.putText(frame, label, (wx - 18, wy + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, shadow_color, 3)
    cv2.putText(frame, label, (wx - 20, wy + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
    for idx, name in FINGER_NAMES.items():
        x = int(hand_landmarks.landmark[idx].x * w)
        y = int(hand_landmarks.landmark[idx].y * h)
        # Shadow effect for finger names
        cv2.putText(frame, name, (x + 6, y - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, shadow_color, 2)
        cv2.putText(frame, name, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

def draw_settings_button(frame, hover=False):
    """Draw the settings button in the top-right corner with enhanced styling."""
    color = (150, 150, 150) if hover else (100, 100, 100)
    bg_color = (45, 45, 45) if hover else (35, 35, 35)
    thickness = 3 if hover else 2
    # Button background with gradient effect
    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), bg_color, -1)
    # Button border with enhanced visibility
    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), color, thickness)
    # Shadow effect
    cv2.rectangle(frame, (BTN_X1 + 2, BTN_Y1 + 2), (BTN_X2 - 1, BTN_Y2 - 1), (20, 20, 20), 1)
    # Button text with shadow
    text_color = (255, 255, 255) if hover else (230, 230, 230)
    cv2.putText(frame, "[ SETTINGS ]", (BTN_X1 + 5, BTN_Y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (50, 50, 50), 2)
    cv2.putText(frame, "[ SETTINGS ]", (BTN_X1 + 4, BTN_Y1 + 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 1)

def draw_settings_ui(frame):
    """Semi-transparent Settings UI Panel with enhanced styling and animations."""
    mx, my = state["mouse_pos"]
    overlay = frame.copy()
    
    # Main Panel Background with border
    panel_x1, panel_y1 = WIDTH//4, HEIGHT//4
    panel_x2, panel_y2 = 3*WIDTH//4, 3*HEIGHT//4
    # Shadow effect
    cv2.rectangle(overlay, (panel_x1 + 3, panel_y1 + 3), (panel_x2 + 3, panel_y2 + 3), (10, 10, 10), -1)
    # Main panel
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (35, 35, 35), -1)
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (220, 220, 220), 3)
    
    # Top accent bar
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y1 + 5), (100, 200, 255), -1)
    
    # Title with shadow
    title_x = panel_x1 + 50
    title_y = panel_y1 + 50
    cv2.putText(overlay, "SETTINGS MENU", (title_x + 2, title_y + 2), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (30, 30, 30), 3)
    cv2.putText(overlay, "SETTINGS MENU", (title_x, title_y), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # Change Folder Button Logic with enhanced styling
    f_rect = (WIDTH//2 - 160, HEIGHT//2 - 50, WIDTH//2 + 160, HEIGHT//2 + 10)
    f_hover = f_rect[0] < mx < f_rect[2] and f_rect[1] < my < f_rect[3]
    f_color = (80, 200, 80) if f_hover else (60, 120, 60)
    f_text_color = (255, 255, 255) if f_hover else (220, 220, 220)
    f_thickness = 3 if f_hover else 2
    # Button shadow
    cv2.rectangle(overlay, (f_rect[0] + 2, f_rect[1] + 2), (f_rect[2] + 2, f_rect[3] + 2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (f_rect[0], f_rect[1]), (f_rect[2], f_rect[3]), f_color, -1)
    cv2.rectangle(overlay, (f_rect[0], f_rect[1]), (f_rect[2], f_rect[3]), (255, 255, 255) if f_hover else (180, 180, 180), f_thickness)
    cv2.putText(overlay, "Change Save Folder", (f_rect[0] + 25, f_rect[1] + 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (30, 30, 30) if f_hover else (20, 20, 20), 2)
    cv2.putText(overlay, "Change Save Folder", (f_rect[0] + 24, f_rect[1] + 37), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, f_text_color, 1)

    # Back Button Logic with enhanced styling
    b_rect = (WIDTH//2 - 70, 3*HEIGHT//4 - 70, WIDTH//2 + 70, 3*HEIGHT//4 - 20)
    b_hover = b_rect[0] < mx < b_rect[2] and b_rect[1] < my < b_rect[3]
    b_color = (200, 80, 80) if b_hover else (120, 60, 120)
    b_text_color = (255, 255, 255) if b_hover else (220, 220, 220)
    b_thickness = 3 if b_hover else 2
    # Button shadow
    cv2.rectangle(overlay, (b_rect[0] + 2, b_rect[1] + 2), (b_rect[2] + 2, b_rect[3] + 2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), b_color, -1)
    cv2.rectangle(overlay, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), (255, 255, 255) if b_hover else (180, 180, 180), b_thickness)
    cv2.putText(overlay, "BACK", (b_rect[0] + 16, b_rect[1] + 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (30, 30, 30), 2)
    cv2.putText(overlay, "BACK", (b_rect[0] + 15, b_rect[1] + 37), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, b_text_color, 1)

    # Fullscreen Toggle Switch - Simple Rectangle Toggle
    toggle_label_x = WIDTH//2 - 250
    toggle_label_y = HEIGHT//2 + 60
    cv2.putText(overlay, "Fullscreen:", (toggle_label_x, toggle_label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (220, 220, 220), 2)
    
    # Toggle switch dimensions - rectangular and compact
    switch_x1 = WIDTH//2 + 20
    switch_y1 = toggle_label_y - 30
    switch_x2 = switch_x1 + 180
    switch_y2 = switch_y1 + 60
    switch_mid = (switch_x1 + switch_x2) // 2
    
    # Check if toggle is hovered
    toggle_hover = (switch_x1 - 10 < mx < switch_x2 + 10 and 
                    switch_y1 - 10 < my < switch_y2 + 10)
    
    # LEFT SIDE - OFF (Red)
    off_color = (120, 40, 40) if toggle_hover and not state["fullscreen_toggle"] else (80, 30, 30)
    cv2.rectangle(overlay, (switch_x1, switch_y1), (switch_mid, switch_y2), off_color, -1)
    cv2.rectangle(overlay, (switch_x1, switch_y1), (switch_mid, switch_y2), (200, 100, 100), 2)
    cv2.putText(overlay, "OFF", (switch_x1 + 10, switch_y1 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 100, 100), 2)
    
    # RIGHT SIDE - ON (Green)
    on_color = (40, 120, 40) if toggle_hover and state["fullscreen_toggle"] else (30, 80, 30)
    cv2.rectangle(overlay, (switch_mid, switch_y1), (switch_x2, switch_y2), on_color, -1)
    cv2.rectangle(overlay, (switch_mid, switch_y1), (switch_x2, switch_y2), (100, 200, 100), 2)
    cv2.putText(overlay, "ON", (switch_x2 - 45, switch_y1 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 200, 100), 2)
    
    # Sliding button indicator
    if state["fullscreen_toggle"]:
        slider_x1 = switch_mid + 3
        slider_x2 = switch_x2 - 3
        slider_color = (100, 255, 100)
    else:
        slider_x1 = switch_x1 + 3
        slider_x2 = switch_mid - 3
        slider_color = (255, 100, 100)
    
    slider_y1 = switch_y1 + 6
    slider_y2 = switch_y2 - 6
    
    # Draw slider bar
    cv2.rectangle(overlay, (slider_x1, slider_y1), (slider_x2, slider_y2), slider_color, -1)
    cv2.rectangle(overlay, (slider_x1, slider_y1), (slider_x2, slider_y2), (255, 255, 255), 2)

    if state["btn_clicked"]:
        if f_hover:
            chosen = open_folder_dialog()
            if chosen: state["screenshots_dir"] = chosen
        elif b_hover:
            state["show_ui"] = False
        elif toggle_hover:
            state["fullscreen_toggle"] = not state["fullscreen_toggle"]
            if state["fullscreen_toggle"]:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        state["btn_clicked"] = False

    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

def open_folder_dialog():
    """Open a folder selection dialog and return the chosen path."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(
        title="Select folder to save screenshots",
        initialdir=state["screenshots_dir"]
    )
    root.destroy()
    return chosen if chosen else None

def imwrite_unicode(path, img):
    """cv2.imwrite wrapper that handles Unicode paths on Windows."""
    ext = Path(path).suffix.lower()
    success, buf = cv2.imencode(ext, img)
    if not success:
        print(f"[ERROR] Encoding failed for: {path}")
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True

def save_screenshot(frame, canvas, canvas_only=False):
    """Save the current view to state['screenshots_dir']."""
    save_dir = state["screenshots_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix    = "drawing" if canvas_only else "snapshot"
    filename  = str(Path(save_dir) / f"{prefix}_{timestamp}.png")
    target = canvas if canvas_only else frame
    result = imwrite_unicode(filename, target)
    if result:
        print(f"[OK] Saved to: {filename}")
    return filename

# ── Mouse callback ─────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    state["mouse_pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["btn_clicked"] = True

# ── Window setup (Fullscreen) ──────────────────────────────────────────────────
WINDOW_NAME = "Hand Drawing"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]

    if not state["show_ui"]:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_landmarks  = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label_raw = results.multi_handedness[i].classification[0].label
                label     = "LEFT" if label_raw == "Left" else "RIGHT"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw_hand_labels(frame, hand_landmarks, label, w, h)

                if label == "LEFT":
                    left_landmarks = hand_landmarks.landmark
                elif label == "RIGHT":
                    lm = hand_landmarks.landmark
                    index  = lm[8].y  < lm[6].y
                    middle = lm[12].y < lm[10].y
                    ring   = lm[16].y < lm[14].y
                    pinky  = lm[20].y < lm[18].y

                    if index and not middle and not ring and not pinky:
                        state["mode"] = "DRAW"
                    elif index and middle and not ring and not pinky:
                        state["mode"] = "FREE"
                    elif index and middle and ring and not pinky:
                        state["mode"] = "ERASE"

        # ── Drawing logic (Left hand) ──
        if left_landmarks:
            if state["mode"] == "DRAW":
                if only_index_up(left_landmarks):
                    draw_point = get_pos(left_landmarks[8], w, h)
                    if state["prev_point"]:
                        cv2.line(canvas, state["prev_point"], draw_point, DRAW_COLOR, DRAW_THICKNESS)
                    state["prev_point"] = draw_point
                    # Enhanced pen indicator with glow
                    cv2.circle(frame, draw_point, 12, (0, 150, 0), 2)
                    cv2.circle(frame, draw_point, 8, DRAW_COLOR, -1)
                    cv2.circle(frame, draw_point, 4, (255, 255, 255), -1)
                else:
                    state["prev_point"] = None
            elif state["mode"] == "FREE":
                state["prev_point"] = None
            elif state["mode"] == "ERASE":
                if is_open_hand(left_landmarks):
                    size   = hand_size(left_landmarks, w, h)
                    center = hand_center(left_landmarks, w, h)
                    cv2.circle(canvas, center, size, (0, 0, 0), -1)
                    # Enhanced eraser indicator with animation
                    cv2.circle(frame, center, size + 5, (100, 100, 255), 2)
                    cv2.circle(frame, center, size, (0, 100, 255), 2)
                    cv2.circle(frame, center, size - 5, (255, 100, 0), 1)
                    state["prev_point"] = None
                elif only_index_up(left_landmarks):
                    draw_point = get_pos(left_landmarks[8], w, h)
                    if state["prev_point"]:
                        cv2.line(canvas, state["prev_point"], draw_point, (0, 0, 0), ERASER_SIZE)
                    state["prev_point"] = draw_point
                    # Enhanced eraser pen with visual feedback
                    cv2.circle(frame, draw_point, ERASER_SIZE // 2 + 3, (100, 100, 255), 2)
                    cv2.circle(frame, draw_point, ERASER_SIZE // 2, (0, 100, 255), 2)
                else:
                    state["prev_point"] = None
        else:
            state["prev_point"] = None

        # Merge canvas
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = canvas[mask > 0]

        # ── HUD with Enhanced Styling ──
        mode_colors = {"DRAW": (0, 255, 100), "FREE": (100, 200, 255), "ERASE": (0, 150, 255)}
        mode_bg_colors = {"DRAW": (20, 80, 20), "FREE": (20, 60, 100), "ERASE": (20, 60, 100)}
        
        # Mode indicator with background panel
        cv2.rectangle(frame, (8, 8), (320, 65), mode_bg_colors[state['mode']], -1)
        cv2.rectangle(frame, (8, 8), (320, 65), mode_colors[state['mode']], 3)
        # Mode text shadow
        cv2.putText(frame, f"MODE: {state['mode']}", (13, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 20), 3)
        cv2.putText(frame, f"MODE: {state['mode']}", (12, 41),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, mode_colors[state["mode"]], 2)

        # Path display with enhanced styling
        display_path = state["screenshots_dir"]
        if len(display_path) > 50: display_path = "..." + display_path[-47:]
        cv2.rectangle(frame, (8, 68), (720, 98), (25, 25, 25), -1)
        cv2.rectangle(frame, (8, 68), (720, 98), (100, 100, 100), 2)
        cv2.putText(frame, f"Save: {display_path}", (14, 88), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, f"Save: {display_path}", (13, 87), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

        # Controls text with background
        controls_text = "RIGHT: 1=Draw | 2=Free | 3=Erase  |  S=Snapshot | D=Drawing | C=Clear | Q=Quit"
        cv2.rectangle(frame, (8, h - 28), (w - 8, h - 2), (25, 25, 25), -1)
        cv2.rectangle(frame, (8, h - 28), (w - 8, h - 2), (80, 80, 80), 2)
        cv2.putText(frame, controls_text, (14, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.48, (80, 80, 80), 2)
        cv2.putText(frame, controls_text, (13, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 200, 200), 1)

        # Settings button interaction
        mx, my  = state["mouse_pos"]
        hovered = BTN_X1 <= mx <= BTN_X2 and BTN_Y1 <= my <= BTN_Y2
        if hovered and state["btn_clicked"]:
            state["show_ui"] = True
            state["btn_clicked"] = False
        draw_settings_button(frame, hover=hovered)

    else:
        # ── UI MODE ──
        draw_settings_ui(frame)

    # Save confirmation flash with enhanced effect
    if state["save_flash"] > 0:
        overlay = frame.copy()
        # Create gradient flash effect
        alpha = state["save_flash"] / 15
        flash_intensity = int(255 * alpha * 0.5)
        cv2.rectangle(overlay, (0, 0), (w, h), (150, 255, 150), -1)
        cv2.addWeighted(overlay, alpha * 0.35, frame, 1 - alpha * 0.35, 0, frame)
        
        # Get text size for proper centering
        text = "SAVED!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Rectangle dimensions and position - centered
        rect_padding = 40
        rect_width = text_size[0] + rect_padding * 2
        rect_height = text_size[1] + rect_padding * 2
        rect_x1 = w // 2 - rect_width // 2
        rect_y1 = h // 2 - rect_height // 2
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        
        # Draw rectangle with proper framing
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 3)
        
        # Place text centered in rectangle with proper alignment
        text_x = w // 2 - text_size[0] // 2
        text_y = h // 2 + text_size[1] // 2
        
        # SAVED! text with shadow and glow
        cv2.putText(frame, text, (text_x + 2, text_y + 2), 
                    font, font_scale, (50, 50, 50), thickness + 1)
        cv2.putText(frame, text, (text_x, text_y), 
                    font, font_scale, (0, 255, 0), thickness)
        state["save_flash"] -= 1

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    elif key == ord('s'):
        save_screenshot(frame, canvas, False)
        state["save_flash"] = 15
    elif key == ord('d'):
        save_screenshot(frame, canvas, True)
        state["save_flash"] = 15

cap.release()
hands.close()
cv2.destroyAllWindows()