#!/usr/bin/env python3
"""
ascii_silhouette.py
Fast person silhouette masking + ASCII output.
Modes: terminal (prints frames of ASCII) or http (Flask MJPEG stream of ASCII images).
"""

import argparse
import threading
import time
import sys
from queue import Queue

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# ---------- Config ----------
CAP_WIDTH = 640   # change to 320 for more speed
CAP_HEIGHT = 480
TARGET_FPS = 20
#ASCII_CHARS = " .,:;i1tfLCG08@"  # from light to dark
ASCII_CHARS = "@#80GCLft1i;:,.  " # from dark to light
FONT_SIZE = 10  # font cell size in px (PIL)
MORPH_KERNEL = 5  # morphological kernel for mask cleaning
MASK_THRESHOLD = 0.5
# ----------------------------

mp_selfie = mp.solutions.selfie_segmentation

def frame_reader(cap, q, target_fps):
    interval = 1.0 / target_fps
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)
        elapsed = time.time() - t0
        to_sleep = interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)


_prev_mask = None
ALPHA = 0.55

def smooth_mask(mask_in, frame_shape):
    """
    mask_in: None or HxW float-like (values in 0..1) returned by MediaPipe
    frame_shape: frame.shape tuple (H, W, ...)
    Returns: HxW float32 mask in 0..1 (never None)
    """
    global _prev_mask
    H, W = frame_shape[:2]

    # If MediaPipe returned nothing, reuse previous mask if available
    if mask_in is None:
        if _prev_mask is None:
            _prev_mask = np.zeros((H, W), dtype=np.float32)
        # ensure returned array matches current frame size
        if _prev_mask.shape != (H, W):
            _prev_mask = cv2.resize(_prev_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        return _prev_mask

    # Convert input to float32 numpy array and resize if needed
    mask_float = np.asarray(mask_in, dtype=np.float32)
    if mask_float.shape != (H, W):
        mask_float = cv2.resize(mask_float, (W, H), interpolation=cv2.INTER_LINEAR)

    # Initialize or update EMA
    if _prev_mask is None:
        _prev_mask = mask_float.copy()
    else:
        if _prev_mask.shape != (H, W):
            _prev_mask = cv2.resize(_prev_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        _prev_mask = ALPHA * _prev_mask + (1.0 - ALPHA) * mask_float

    # clamp to [0,1]
    np.clip(_prev_mask, 0.0, 1.0, out=_prev_mask)
    return _prev_mask

"""
def refine_mask(mask_float):
    # mask_float: HxW float32 in 0..1
    mask = (mask_float * 255).astype(np.uint8)
    # morphological open then close for noise removal and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # optionally smooth edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask
"""

def refine_mask(mask_float):
    """
    mask_float: HxW float32 in 0..1
    Returns: uint8 mask 0..255, refined for tight silhouette.
    """
    mask = (mask_float * 255).astype(np.uint8)

    # 1. Slight dilation to close gaps (fill holes between arm and torso)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 2. Optional edge smoothing: small blur then hard threshold
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    return mask

def ascii_from_frame(frame, mask, cols=120, contrast=1.4, brightness=-40):
    """
    frame: BGR uint8
    mask: uint8 (0..255)
    contrast: >1 increases contrast
    brightness: added after contrast (negative = darker)
    """
    h, w = frame.shape[:2]
    cell_w = w / cols
    cell_h = cell_w * 2.0
    rows = int(h / cell_h)
    if rows < 1: rows = 1

    # Resize frame and mask
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small_mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_LINEAR)

    # --- contrast + brightness correction ---
    small_gray = np.clip(small_gray.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    # Build ASCII
    chars = ASCII_CHARS
    n_chars = len(chars)
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            if small_mask[r, c] < 50:
                row_chars.append(" ")
            else:
                lum = small_gray[r, c] / 255.0
                idx = int((1.0 - lum) * (n_chars - 1))
                row_chars.append(chars[idx])
        lines.append("".join(row_chars))
    return "\n".join(lines)


def pil_image_from_ascii(ascii_text, font=None, bg=(0,0,0), fg=(255,255,255)):
    lines = ascii_text.splitlines()
    if font is None:
        font = ImageFont.load_default()
    # estimate size
    max_width = max((font.getsize(line)[0] for line in lines), default=1)
    total_height = sum((font.getsize(line)[1] for line in lines), 0)
    img = Image.new("RGB", (max_width, total_height), color=bg)
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=fg)
        y += font.getsize(line)[1]
    return img


# ---------- Terminal mode ----------
def run_terminal(cap):
    q = Queue(maxsize=2)
    t = threading.Thread(target=frame_reader, args=(cap, q, TARGET_FPS), daemon=True)
    t.start()

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        try:
            while True:
                if q.empty():
                    time.sleep(0.001)
                    continue
                frame = q.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = seg.process(frame_rgb)
                if res.segmentation_mask is None:
                    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                else:
                    mask_float = smooth_mask(res.segmentation_mask, frame.shape)
                    mask = refine_mask(mask_float) 
                    #mask = refine_mask(res.segmentation_mask)
                
                ascii_text = ascii_from_frame(frame, mask, cols=120, contrast=1.4, brightness=-40)
                # Clear terminal and print
                sys.stdout.write("\x1b[H\x1b[2J")
                sys.stdout.write(ascii_text)
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass


# ---------- HTTP mode ----------
from flask import Flask, Response, render_template_string
app = Flask(__name__)
frame_queue = Queue(maxsize=2)


def run_http_server(cap, host="0.0.0.0", port=5000):
    # start reader thread
    t = threading.Thread(target=frame_reader, args=(cap, frame_queue, TARGET_FPS), daemon=True)
    t.start()

    @app.route("/")
    def index():
        return render_template_string("""
            <html>
                <head><title>ASCII Silhouette</title></head>
                <body style="background:black; color:white;">
                  <h3 style="color:lightgreen">ASCII silhouette stream</h3>
                  <img src="/video_feed" />
                </body>
            </html>
        """)

    def gen():
        font = ImageFont.load_default()
        with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
            while True:
                if frame_queue.empty():
                    time.sleep(0.001)
                    continue
                frame = frame_queue.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = seg.process(frame_rgb)
                if res.segmentation_mask is None:
                    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                else:
                    mask = refine_mask(res.segmentation_mask)
                ascii_text = ascii_from_frame(frame, mask, cols=120)
                img = pil_image_from_ascii(ascii_text, font=font)
                # convert to JPEG bytes
                buf = cv2.imencode('.jpg', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf + b'\r\n')
                # frame pacing
                time.sleep(1.0 / TARGET_FPS)

    @app.route("/video_feed")
    def video_feed():
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # run in same thread (blocking)
    app.run(host=host, port=port, threaded=True)


# ---------- Main ----------
def main():
    global CAP_WIDTH, CAP_HEIGHT, TARGET_FPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["terminal","http"], default="terminal")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=CAP_WIDTH)
    parser.add_argument("--height", type=int, default=CAP_HEIGHT)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    args = parser.parse_args()

    CAP_WIDTH, CAP_HEIGHT, TARGET_FPS = args.width, args.height, args.fps

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("Cannot open camera", file=sys.stderr)
        return

    if args.mode == "terminal":
        run_terminal(cap)
    else:
        run_http_server(cap)

    cap.release()


if __name__ == "__main__":
    main()

