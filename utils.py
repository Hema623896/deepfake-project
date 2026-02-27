import cv2
import os
import numpy as np


def extract_frames(video_path, max_frames=8, resize=None):
    # By default return frames in their original size (resize=None). Model preprocessing will handle scaling.
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    step = max(1, total // max_frames) if total > 0 else 1
    idx = 0
    grabbed = 0
    while grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
            grabbed += 1
        idx += 1
    cap.release()
    if len(frames) == 0 and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        cap.release()
    return frames
