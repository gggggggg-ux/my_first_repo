import cv2
import numpy as np
from ultralytics import YOLO

def load_bar_model(model_path):
    return YOLO(model_path)

def detect_bar_and_pose(model, frame, conf=0.5):
    results = model(frame, conf=conf, verbose=False)
    bar_info = None
    pose_kpts = None
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bar_info = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "bar_y": (y1 + y2) / 2
                    }
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            pose_kpts = r.keypoints.xy[0].cpu().numpy()
    return bar_info, pose_kpts

def draw_bar(frame, bar_info, color=(0,0,255)):
    if bar_info is None:
        return frame
    x1 = bar_info["x1"]
    y1 = bar_info["y1"]
    x2 = bar_info["x2"]
    y2 = bar_info["y2"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    return frame

def draw_bar_pose(frame, kpts):
    if kpts is None:
        return frame
    for x, y in kpts:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    return frame
