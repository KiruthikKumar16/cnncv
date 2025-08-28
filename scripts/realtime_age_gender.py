import os
from pathlib import Path
import time
from typing import Optional, Tuple
import argparse

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


SMOOTHING_BETA = float(os.getenv("SMOOTH_BETA", "0.9"))
SMOOTHING_ENABLED = True
DISPLAY_SCALE = float(os.getenv("DISPLAY_SCALE", "1.0"))
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "0"))  # 0 = leave default
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "0"))  # 0 = leave default
CAM_INDEX_ENV = os.getenv("CAM_INDEX")


def _put_text_box(img: np.ndarray, text: str, x: int, y_baseline: int,
                  font_scale: float = 0.6, thickness: int = 2,
                  text_color = (0, 255, 0), box_color = (0, 0, 0)) -> int:
    if not text:
        return y_baseline
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 2, min(y_baseline, img.shape[0] - 2))
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline), box_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    return y


def _draw_labels_block(img: np.ndarray, box: tuple, label_top: str, label_bottom: str,
                       color = (0, 255, 0)) -> None:
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    gap = 4
    margin = 6

    (w_top, h_top), base_top = cv2.getTextSize(label_top or " ", font, font_scale, thickness)
    (w_bot, h_bot), base_bot = cv2.getTextSize(label_bottom or " ", font, font_scale, thickness)
    block_w = max(w_top, w_bot)
    x = min(max(0, x1), img.shape[1] - block_w - 4)

    need_h = (h_top + (h_bot + gap if label_bottom else 0))

    # Prefer above the box if there's room for both lines
    if y1 - margin - need_h - 2 >= 0:
        y_bottom = y1 - margin
        if label_bottom:
            _put_text_box(img, label_bottom, x, y_bottom)
            y_top = y_bottom - (h_bot + gap)
            _put_text_box(img, label_top, x, y_top)
        else:
            _put_text_box(img, label_top, x, y_bottom)
        return

    # Else, try drawing inside the box near the top-left
    if (y2 - y1) - margin - need_h - 2 >= 0:
        y_top = y1 + margin + h_top
        _put_text_box(img, label_top, x, y_top)
        if label_bottom:
            y_bottom = y_top + h_bot + gap
            _put_text_box(img, label_bottom, x, y_bottom)
        return

    # Else, draw below the box
    y_top = y2 + margin + h_top
    _put_text_box(img, label_top, x, y_top)
    if label_bottom:
        y_bottom = y_top + h_bot + gap
        _put_text_box(img, label_bottom, x, y_bottom)


def _draw_single_line_label(img: np.ndarray, box: tuple, text: str,
                            font_scale: float = 0.6, thickness: int = 2,
                            text_color = (0, 255, 0), box_color = (0, 0, 0)) -> None:
    if not text:
        return
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 6
    # Prefer above
    y = y1 - margin
    x = min(max(0, x1), max(0, img.shape[1] - tw - 4))
    if y - th - 2 < 0:
        # Try below
        y = y2 + margin + th
        if y + baseline >= img.shape[0]:
            # Draw inside near top-left of the box
            y = y1 + margin + th
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline), box_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def load_face_detector() -> Optional[Tuple[cv2.dnn_Net, int, int]]:
    proto = MODELS_DIR / "face" / "deploy.prototxt"
    model = MODELS_DIR / "face" / "res10_300x300_ssd_iter_140000.caffemodel"
    if proto.exists() and model.exists():
        net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        return net, 300, 300
    return None


def load_caffe_net(proto_path: Path, model_path: Path) -> Optional[cv2.dnn_Net]:
    if proto_path.exists() and model_path.exists():
        return cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    return None


def main():
    parser = argparse.ArgumentParser(description="Real-time Age & Gender detection")
    parser.add_argument("--scale", type=float, default=DISPLAY_SCALE, help="Display scaling factor (e.g., 1.5)")
    parser.add_argument("--cam-width", type=int, default=CAM_WIDTH, help="Request camera width (pixels)")
    parser.add_argument("--cam-height", type=int, default=CAM_HEIGHT, help="Request camera height (pixels)")
    parser.add_argument("--cam-index", type=int, default=int(CAM_INDEX_ENV) if CAM_INDEX_ENV is not None else None, help="Force camera index")
    args = parser.parse_args()
    # Load face detector (DNN if available, else fallback to Haar cascade)
    face_dnn = load_face_detector()
    haar_cascade = None
    if face_dnn is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            print("No face detector found. Please download models or install OpenCV data.")
            return

    # Load age and gender nets if available
    age_net = load_caffe_net(MODELS_DIR / "age" / "age_deploy.prototxt", MODELS_DIR / "age" / "age_net.caffemodel")
    gender_net = load_caffe_net(MODELS_DIR / "gender" / "gender_deploy.prototxt", MODELS_DIR / "gender" / "gender_net.caffemodel")

    if age_net is None or gender_net is None:
        print("Warning: Age/Gender models not found. The app will still run but show 'N/A'.")

    age_list = [
        "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ]
    gender_list = ["Male", "Female"]

    mean_vals = (78.4263377603, 87.7689143744, 114.895847746)  # BGR mean for Caffe models

    # Try multiple backends and indices for better Windows compatibility
    cap = None
    preferred_indices = ([args.cam_index] if args.cam_index is not None else [0, 1, 2])
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        for idx in preferred_indices:
            test = cv2.VideoCapture(idx, backend)
            if test.isOpened():
                cap = test
                break
            else:
                test.release()
        if cap is not None:
            break
    if not cap.isOpened():
        print("Unable to open webcam.")
        return

    # Try to set capture resolution if requested
    if args.cam_width and args.cam_height and args.cam_width > 0 and args.cam_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    print("Press 'q' to quit.")
    fps_avg = None
    last_time = time.time()

    def softmax(logits: np.ndarray) -> np.ndarray:
        max_logit = np.max(logits)
        exps = np.exp(logits - max_logit)
        sum_exps = np.sum(exps)
        return exps / sum_exps if sum_exps > 0 else np.zeros_like(logits)

    def l2(a: tuple, b: tuple) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def ema_update(prev: np.ndarray, new: np.ndarray, beta: float) -> np.ndarray:
        if prev is None:
            return new.copy()
        return beta * prev + (1.0 - beta) * new

    # Simple nearest-neighbor tracking for smoothing across frames
    tracks = []  # each: {center:(cx,cy), gender:np.ndarray(2), age:np.ndarray(8), last_seen:float}

    win_name = "Age & Gender (q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    window_sized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for a natural webcam view
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        faces = []  # List of dicts: {"box": (x1,y1,x2,y2), "conf": Optional[float]}

        if face_dnn is not None:
            net, in_w, in_h = face_dnn
            blob = cv2.dnn.blobFromImage(frame, 1.0, (in_w, in_h), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < 0.7:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    faces.append({"box": (x1, y1, x2, y2), "conf": confidence})
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, ww, hh) in rects:
                faces.append({"box": (x, y, x + ww, y + hh), "conf": None})

        now_ts = time.time()
        max_assign_dist = max(w, h) * 0.25

        for face in faces:
            (x1, y1, x2, y2) = face["box"]
            det_conf = face["conf"]
            face_roi = frame[y1:y2, x1:x2]

            label_top = ""
            label_bottom = ""
            if age_net is not None and gender_net is not None and face_roi.size:
                blob = cv2.dnn.blobFromImage(
                    image=face_roi,
                    scalefactor=1.0,
                    size=(227, 227),
                    mean=mean_vals,
                    swapRB=False,
                    crop=False,
                )

                # Gender
                gender_net.setInput(blob)
                gender_logits = gender_net.forward().flatten()
                gender_probs_raw = softmax(gender_logits)

                # Age
                age_net.setInput(blob)
                age_logits = age_net.forward().flatten()
                age_probs_raw = softmax(age_logits)

                # Temporal smoothing via EMA with simple track association
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                assigned = False
                if SMOOTHING_ENABLED and len(tracks) > 0:
                    # find nearest track within threshold
                    dists = [l2((cx, cy), t["center"]) for t in tracks]
                    if len(dists) > 0:
                        j = int(np.argmin(np.array(dists)))
                        if dists[j] <= max_assign_dist:
                            tracks[j]["gender"] = ema_update(tracks[j]["gender"], gender_probs_raw, SMOOTHING_BETA)
                            tracks[j]["age"] = ema_update(tracks[j]["age"], age_probs_raw, SMOOTHING_BETA)
                            tracks[j]["center"] = (cx, cy)
                            tracks[j]["last_seen"] = now_ts
                            gender_probs = tracks[j]["gender"]; age_probs = tracks[j]["age"]
                            assigned = True
                if not assigned:
                    tracks.append({
                        "center": (cx, cy),
                        "gender": gender_probs_raw.copy(),
                        "age": age_probs_raw.copy(),
                        "last_seen": now_ts,
                    })
                    gender_probs = gender_probs_raw; age_probs = age_probs_raw

                # Renormalize after EMA
                gsum = float(np.sum(gender_probs)); asumeps = float(np.sum(age_probs))
                if gsum > 0:
                    gender_probs = gender_probs / gsum
                if asumeps > 0:
                    age_probs = age_probs / asumeps

                gender_idx = int(np.argmax(gender_probs))
                gender = gender_list[gender_idx]
                gender_p = float(gender_probs[gender_idx])
                age_idx = int(np.argmax(age_probs))
                age = age_list[age_idx]
                age_p = float(age_probs[age_idx])

                # Build labels with probabilities and face detection confidence
                if det_conf is not None:
                    label_top = f"{gender} {gender_p*100:.0f}%  |  Face {det_conf*100:.0f}%"
                else:
                    label_top = f"{gender} {gender_p*100:.0f}%"
                label_bottom = f"Age: {age} {age_p*100:.0f}%"
            else:
                label_top = "N/A"
                label_bottom = ""

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Compose single-line label to avoid clashes and support groups
            single_line = label_top
            if label_bottom:
                single_line = f"{single_line} | {label_bottom}"
            _draw_single_line_label(frame, (x1, y1, x2, y2), single_line)

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        fps_avg = fps if fps_avg is None else (0.9 * fps_avg + 0.1 * fps)
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Cleanup stale tracks
        if len(tracks) > 0:
            tracks = [t for t in tracks if (now - t["last_seen"]) < 1.0]

        # Scale display if requested
        scale = args.scale if args.scale and args.scale > 0 else 1.0
        display_frame = frame if abs(scale - 1.0) < 1e-3 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if not window_sized:
            cv2.resizeWindow(win_name, display_frame.shape[1], display_frame.shape[0])
            window_sized = True

        cv2.imshow(win_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


