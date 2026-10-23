

from __future__ import annotations

from pathlib import Path
import os
import sys
import time
from urllib.request import urlretrieve


try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        f"Missing Python package: {exc}.\n"
        f"Install for this Python ({sys.executable}):\n"
        f'  "{sys.executable}" -m pip install opencv-python numpy mediapipe==0.10.33'
    )

NEEDED_CV2 = ("VideoCapture", "imshow", "resize", "cvtColor", "waitKey", "putText")
missing_cv2 = [name for name in NEEDED_CV2 if not hasattr(cv2, name)]
if missing_cv2:
    raise SystemExit(
        "This Python imported a broken/incomplete cv2 package (missing "
        + ", ".join(missing_cv2)
        + ").\nPython: " + sys.executable
        + "\ncv2 loaded from: " + str(getattr(cv2, "__file__", "unknown"))
        + "\nUse a clean Windows Python virtual environment; do not run it "
          "with the MSYS2 UCRT64 Python or Code Runner."
    )

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "images"
MODEL_PATH = HERE / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
FACE_MODEL_PATH = HERE / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
GESTURE_IMAGE = {
    "Thumb_Up": "cat_thumbs_up.jpg",
    "Pointing_Up": "cat_glasses.jpg",
    "Shush": "cat_shush.jpg",
    "Flashlight": "cat_flashlight.jpg",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PANEL_WIDTH = 560
PANEL_HEIGHT = 420
WINDOW_NAME = "Hand Gesture Recognition - OpenCV"
BG = (24, 24, 24)

FLASH_CALIBRATION_FRAMES = 6
FLASH_FRAMES_NEEDED = 2
FLASH_HOLD_SECONDS = 3.5

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def get_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MediaPipe Tasks could not load. In VS Code PowerShell, use a "
            "regular Windows Python (preferably Python 3.12), then install:\n"
            "  python -m pip install opencv-python numpy mediapipe==0.10.33\n"
            f"Details: {exc}"
        ) from exc
    return mp, mp_python, vision


def ensure_model(path, url, description):
    if path.is_file() and path.stat().st_size > 100_000:
        return
    print(f"Downloading {description} model (first run only)...")
    temp = path.with_suffix(".task.part")
    try:
        urlretrieve(url, temp)
        if temp.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temp.replace(path)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {description} model: {exc}\n"
            "Connect to the internet for the first run, or download:\n"
            f"{url}\nSave it as '{path.name}' in the GestureRecognition folder."
        ) from exc


def pixel_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-8:
        return pixel_distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx +
                                (point[1] - start[1]) * dy) / length2))
    nearest = (start[0] + t * dx, start[1] + t * dy)
    return pixel_distance(point, nearest)


def is_shushing(hands, faces, frame_width, frame_height, recognized_gesture):
    if not hands or not faces:
        return False

    hand = hands[0]
    face = faces[0]

    def hand_xy(i):
        return (hand[i].x * frame_width, hand[i].y * frame_height)

    def face_xy(i):
        return (face[i].x * frame_width, face[i].y * frame_height)

    upper_lip, lower_lip = face_xy(13), face_xy(14)
    mouth = ((upper_lip[0] + lower_lip[0]) / 2,
             (upper_lip[1] + lower_lip[1]) / 2)
    face_width = pixel_distance(face_xy(234), face_xy(454))
    if face_width < 30:
        return False

    tip = hand_xy(8)
    dip = hand_xy(7)
    pip = hand_xy(6)
    mcp = hand_xy(5)

    if not (tip[1] < pip[1] - 5 and
            pip[1] < mcp[1] - 3 and
            abs(tip[0] - pip[0]) <= max(18, (pip[1] - tip[1]) * 1.15)):
        return False

    curled = 0
    wrist = hand_xy(0)
    for finger_tip, finger_pip in [(12, 10), (16, 14), (20, 18)]:
        if pixel_distance(hand_xy(finger_tip), wrist) < \
                1.08 * pixel_distance(hand_xy(finger_pip), wrist):
            curled += 1
    if recognized_gesture != "Pointing_Up" and curled < 2:
        return False

    near_lips = point_segment_distance(mouth, pip, tip)
    horizontal_gap = abs(mouth[0] - (tip[0] + pip[0]) / 2)
    return (near_lips <= max(12.0, face_width * 0.11) and
            horizontal_gap <= face_width * 0.17 and
            pixel_distance(mouth, mcp) <= face_width * 0.95)


class FlashlightDetector:

    def __init__(self):
        self.reset()

    def reset(self):
        self.base_gray = None
        self.calibration_count = 0
        self.streak = 0
        self.armed = True
        self.normal_frames = 0
        self.debug = "Light: calibrating (keep flashlight OFF)"

    def update(self, frame):
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        current_mean = float(np.mean(gray))

        if self.base_gray is None:
            self.base_gray = gray.astype(np.float32)
            self.calibration_count = 1
            return False

        if self.calibration_count < FLASH_CALIBRATION_FRAMES:
            self.base_gray = (self.base_gray * self.calibration_count + gray) / (
                self.calibration_count + 1)
            self.calibration_count += 1
            self.debug = f"Light: calibrating {self.calibration_count}/{FLASH_CALIBRATION_FRAMES}"
            return False

        base = self.base_gray
        positive_change = np.maximum(gray.astype(np.float32) - base, 0.0)
        mean_rise = current_mean - float(np.mean(base))

        
        newly_white = (gray >= 250) & (positive_change >= 90)
        hot_pixels = int(np.count_nonzero(newly_white))
        
        patch_rise = float(cv2.blur(positive_change, (7, 7)).max())

        
        global_light = mean_rise >= 45 and current_mean >= 110
        focused_light = hot_pixels >= 200 and patch_rise >= 80
        flash = global_light or focused_light 

        self.debug = (f"Light: {'ON' if flash else 'ready'}  "
                      f"rise={mean_rise:+.0f}  bright_pixels={hot_pixels}  "
                      f"spot={patch_rise:.0f}")

        if flash:
            self.streak += 1
            self.normal_frames = 0
        else:
            self.streak = 0
            self.normal_frames += 1
            
            self.base_gray = 0.99 * base + 0.01 * gray
            if self.normal_frames >= 6:
                self.armed = True

        if self.armed and self.streak >= FLASH_FRAMES_NEEDED:
            self.armed = False
            return True
        return False


def fit_panel(image):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot resize empty image")
    scale = min(PANEL_WIDTH / w, PANEL_HEIGHT / h)
    width = max(1, round(w * scale))
    height = max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (width, height), interpolation=interpolation)
    out = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), BG, dtype=np.uint8)
    start_x = (PANEL_WIDTH - width) // 2
    start_y = (PANEL_HEIGHT - height) // 2
    out[start_y:start_y+height, start_x:start_x+width] = scaled
    return out


def load_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    pictures = {}
    for path in sorted(IMAGE_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            picture = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if picture is not None:
                pictures[path.name] = fit_panel(picture)
            else:
                print("Skipping unreadable image:", path.name)
        except (OSError, ValueError, cv2.error) as exc:
            print("Skipping", path.name, exc)
    return pictures


def draw_landmarks(frame, hands):
    h, w = frame.shape[:2]
    for hand in hands:
        xy = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, xy[a], xy[b], (0, 180, 255), 2, cv2.LINE_AA)
        for point in xy:
            cv2.circle(frame, point, 3, (5, 250, 110), -1, cv2.LINE_AA)


def overlay(panel, title, footer=None):
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 40), (14, 14, 14), -1)
    cv2.putText(panel, title[:48], (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if footer:
        y = PANEL_HEIGHT - 36
        cv2.rectangle(panel, (0, y), (PANEL_WIDTH, PANEL_HEIGHT), (14, 14, 14), -1)
        cv2.putText(panel, footer[:57], (12, PANEL_HEIGHT - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (236, 236, 236), 1,
                    cv2.LINE_AA)


def open_camera():
    options = [(0, cv2.CAP_ANY), (1, cv2.CAP_ANY)]
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        options.insert(0, (0, cv2.CAP_DSHOW))
    for index, backend in options:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"Webcam opened (device {index}).")
                return cap, frame
        cap.release()
    raise RuntimeError(
        "Webcam not available. Close other camera apps and check "
        "Windows Settings > Privacy & security > Camera."
    )


def draw_display(frame, pictures, image_name, gesture_label, confidence):

    left = fit_panel(frame)

    if image_name in pictures:
        right = pictures[image_name].copy()
    else:
        right = np.full(
            (PANEL_HEIGHT, PANEL_WIDTH, 3),
            BG,
            dtype=np.uint8
        )

    return np.hstack((left, right))

def main():
    mp, mp_python, vision = get_mediapipe()
    ensure_model(MODEL_PATH, MODEL_URL, "gesture recognition")
    ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarks")
    pictures = load_images()
    names = list(pictures)
    print("Found cat pictures:", ", ".join(names) if names else "NONE")
    print("Thumbs-up -> thumbs-up cat; pointing up -> glasses cat; finger on lips -> shushing cat")
    print("Shush uses a finger-to-lips proximity heuristic; detection may vary with camera angle.")
    print("Shine a phone flashlight into the webcam to show cat_flashlight.jpg (sudden light change).")
    if GESTURE_IMAGE["Flashlight"] not in pictures:
        print("WARNING: images/cat_flashlight.jpg was NOT loaded. Copy it into the images folder!")
    else:
        print("Flashlight cat image found. Press F in the camera window to test it manually.")
    print("Start with flashlight OFF for 2 seconds; then aim it at the webcam for 1-2 seconds.")
    print("Click the OpenCV window. F: test cat  B: recalibrate light  N/P: browse  Q/ESC: quit")

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = None
    try:
        with (vision.GestureRecognizer.create_from_options(options) as recognizer,
              vision.FaceLandmarker.create_from_options(face_options) as face_tracker):
            camera, frame = open_camera()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, PANEL_WIDTH * 2, PANEL_HEIGHT)
            selected = GESTURE_IMAGE["Thumb_Up"] if GESTURE_IMAGE["Thumb_Up"] in pictures else (names[0] if names else None)
            candidate, consecutive = None, 0
            last_timestamp = -1
            last_selection_time = 0.0
            failed_reads = 0
            flashlight_detector = FlashlightDetector()
            flash_label_until = 0.0
            flash_hold_until = 0.0
            while True:
                
                flipped = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(rgb))
                timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
                last_timestamp = timestamp
                result = recognizer.recognize_for_video(mp_frame, timestamp)
                face_result = face_tracker.detect_for_video(mp_frame, timestamp)

                label, confidence = "None", 0.0
                if result.gestures and result.gestures[0]:
                    guess = result.gestures[0][0]
                    label, confidence = guess.category_name, float(guess.score)
                if result.hand_landmarks:
                    draw_landmarks(flipped, result.hand_landmarks)

                
                h, w = flipped.shape[:2]
                shush = is_shushing(
                    result.hand_landmarks, face_result.face_landmarks,
                    w, h, label)
                if shush:
                    label = "Shush"

                
                now = time.monotonic()
                saw_flash = flashlight_detector.update(flipped)
                if saw_flash and GESTURE_IMAGE["Flashlight"] in pictures:
                    selected = GESTURE_IMAGE["Flashlight"]
                    candidate, consecutive = None, 0
                    flash_label_until = now + FLASH_HOLD_SECONDS
                    flash_hold_until = now + FLASH_HOLD_SECONDS
                    print("Detected bright flashlight ->", selected)

                
                if now >= flash_hold_until:
                    match = GESTURE_IMAGE.get(label) if (shush or confidence >= 0.55) else None
                    if match is not None and match in pictures:
                        if candidate == match:
                            consecutive += 1
                        else:
                            candidate, consecutive = match, 1
                        if (consecutive >= 6 and selected != match and
                                now - last_selection_time >= 0.4):
                            selected = match
                            last_selection_time = now
                            print("Recognized:", label, "->", selected)
                    else:
                        candidate, consecutive = None, 0

                if now < flash_label_until:
                    label, confidence = "Flashlight", 1.0

                display = draw_display(flipped, pictures, selected, label, confidence)
                
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("n"), ord("N"), ord("p"), ord("P")) and names:
                    current_idx = names.index(selected) if selected in names else 0
                    diff = 1 if key in (ord("n"), ord("N")) else -1
                    selected = names[(current_idx + diff) % len(names)]
                    candidate, consecutive = None, 0
                elif key in (ord("f"), ord("F")):
                    if GESTURE_IMAGE["Flashlight"] in pictures:
                        selected = GESTURE_IMAGE["Flashlight"]
                        flash_label_until = time.monotonic() + FLASH_HOLD_SECONDS
                        flash_hold_until = flash_label_until
                        print("Manual flashlight image test: OK")
                    else:
                        print("ERROR: images/cat_flashlight.jpg was not found or could not be read.")
                elif key in (ord("b"), ord("B")):
                    flashlight_detector.reset()
                    print("Flashlight detector recalibrated. Keep light OFF for 2 seconds.")
                elif key in (ord("r"), ord("R")):
                    pictures = load_images()
                    names = list(pictures)
                    if selected not in pictures:
                        selected = names[0] if names else None
                    print(f"Reloaded {len(names)} images")

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                ok, next_frame = camera.read()
                if ok and next_frame is not None and next_frame.size:
                    frame = next_frame
                    failed_reads = 0
                else:
                    failed_reads += 1
                    if failed_reads > 30:
                        print("Camera stopped delivering frames.")
                        break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, cv2.error, ValueError, OSError) as exc:
        print("\nERROR:", exc, file=sys.stderr)
        sys.exit(1)