import cv2
import numpy as np
from ultralytics import YOLO

def load_bar_model(model_path):
    return YOLO(model_path)

def detect_bar_and_pose(model, frame, conf=0.6):
    results = model(frame, conf=conf, verbose=False)[0]
    bar_info = None
    bar_kpts = None

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        bar_boxes = boxes[cls == 0]
        if len(bar_boxes) > 0:
            idx = np.argmin(bar_boxes[:, 1])
            x1, y1, x2, y2 = bar_boxes[idx]
            bar_y = (y1 + y2) / 2
            bar_info = {
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "bar_y": bar_y
            }

    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        bar_kpts = results.keypoints.xy[0].cpu().numpy()

    return bar_info, bar_kpts

def draw_bar(frame, bar_info, color=(0,255,255)):
    if bar_info is None:
        return frame
    x1, y1, x2, y2 = bar_info["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.line(frame, (0, int(bar_info["bar_y"])),
             (frame.shape[1], int(bar_info["bar_y"])), color, 1)
    return frame

def draw_bar_pose(frame, bar_kpts, color=(255,0,255)):
    if bar_kpts is None:
        return frame
    for x, y in bar_kpts:
        if x > 5 and y > 5:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    pairs = [(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
    for i,j in pairs:
        if bar_kpts[i][1]>1 and bar_kpts[j][1]>1:
            cv2.line(frame, (int(bar_kpts[i][0]),int(bar_kpts[i][1])),
                     (int(bar_kpts[j][0]),int(bar_kpts[j][1])), color, 1)
    return frame