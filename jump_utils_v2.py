import cv2
import numpy as np
import math
from scipy import signal

# YOLO-Pose 关键点索引
KP_IND = {
    'sh_l':5, 'sh_r':6,
    'hip_l':11, 'hip_r':12,
    'knee_l':13, 'knee_r':14,
    'ankle_l':15, 'ankle_r':16
}

KP_HIST = []
MAX_HIST = 1

def smooth_kpts(kpts):
    global KP_HIST
    if kpts is None:
        KP_HIST.clear()
        return None
    KP_HIST.append(kpts.copy())
    if len(KP_HIST) > MAX_HIST:
        KP_HIST.pop(0)
    return np.mean(np.array(KP_HIST), axis=0)

def calc_center(kpts):
    if kpts is None:
        return -1
    sh = (kpts[5] + kpts[6]) / 2
    hip = (kpts[11] + kpts[12]) / 2
    return sh[1] * 0.3 + hip[1] * 0.7

def calc_angle(a,b,c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    mod_ba = np.linalg.norm(ba)
    mod_bc = np.linalg.norm(bc)
    if mod_ba==0 or mod_bc==0:
        return 0.0
    cos = np.dot(ba, bc)/(mod_ba*mod_bc)
    cos = max(min(cos,1.0),-1.0)
    return math.degrees(math.acos(cos))

def extract_all_frames(video_path, model, peak_frame):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_kpts = []
    all_center = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx < peak_frame - 20:
            conf = 0.30
        else:
            conf = 0.40
        res = model(frame, conf=conf, verbose=False)[0]
        if res.keypoints and len(res.keypoints.xy) > 0:
            raw_kp = res.keypoints.xy[0].cpu().numpy()
            kpts = smooth_kpts(raw_kp)
            center_y = calc_center(kpts)
            all_kpts.append(kpts)
            all_center.append(center_y)
        else:
            all_kpts.append(None)
            all_center.append(-1)
        idx += 1
    cap.release()
    return all_kpts, all_center, fps, w, h, total

def smooth_center_ys(center_ys, window=5):
    ys = np.array(center_ys, dtype=float)
    ys[ys < 0] = np.nan
    if np.all(np.isnan(ys)):
        return center_ys
    xs = np.arange(len(ys))
    ys[np.isnan(ys)] = np.interp(xs[np.isnan(ys)], xs[~np.isnan(ys)], ys[~np.isnan(ys)])
    if window >= 3 and window % 2 == 1:
        ys = signal.medfilt(ys, kernel_size=window)
    ys = np.maximum(ys, 0)
    return ys.tolist()

def convert_to_math_coords(center_ys):
    valid = [y for y in center_ys if y >= 0]
    if not valid:
        return []
    min_y = min(valid)
    max_y = max(valid)
    if max_y - min_y < 1e-3:
        return [50.0] * len(center_ys)
    norm = []
    for y in center_ys:
        if y < 0:
            norm.append(0)
        else:
            h = 100.0 * (1.0 - (y - min_y) / (max_y - min_y))
            norm.append(max(h, 0))
    return norm

def find_peak_frame(norm_heights):
    if not norm_heights:
        return -1
    return int(np.argmax(norm_heights))

def compute_last_step_angles(kpts, takeoff_leg):
    if kpts is None:
        return None, None
    if takeoff_leg == "left":
        hip, knee, ankle = kpts[11], kpts[13], kpts[15]
    else:
        hip, knee, ankle = kpts[12], kpts[14], kpts[16]
    knee_angle = calc_angle(hip, knee, ankle)
    sh_c = (kpts[5] + kpts[6]) / 2
    hip_c = (kpts[11] + kpts[12]) / 2
    dx = sh_c[0] - hip_c[0]
    dy = sh_c[1] - hip_c[1]
    vx, vy = 0, -1
    dot = dx * vx + dy * vy
    mod_body = math.sqrt(dx**2 + dy**2)
    if mod_body == 0:
        trunk_angle = 0.0
    else:
        cos_angle = dot / mod_body
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        trunk_angle = math.degrees(math.acos(cos_angle))
    return round(trunk_angle, 1), round(knee_angle, 1)

def draw_pose(frame, kpts, config, frame_idx, target_frame, range_frame=12):
    if kpts is None:
        return frame
    for x, y in kpts:
        if x > 5 and y > 5:
            cv2.circle(frame, (int(x), int(y)), 3, config.COLOR_POINT, -1)
    if abs(frame_idx - target_frame) <= range_frame:
        sh_c = (int((kpts[5][0] + kpts[6][0])/2), int((kpts[5][1] + kpts[6][1])/2))
        hip_c = (int((kpts[11][0] + kpts[12][0])/2), int((kpts[11][1] + kpts[12][1])/2))
        cv2.line(frame, sh_c, hip_c, (0, 255, 255), 2)
        if config.TAKEOFF_LEG == "left":
            h, k, a = kpts[11], kpts[13], kpts[15]
        else:
            h, k, a = kpts[12], kpts[14], kpts[16]
        cv2.line(frame, (int(h[0]), int(h[1])), (int(k[0]), int(k[1])), (255, 100, 0), 2)
        cv2.line(frame, (int(k[0]), int(k[1])), (int(a[0]), int(a[1])), (255, 100, 0), 2)
    return frame

def draw_last_step_overlay(frame, kpts, trunk_angle, knee_angle, takeoff_leg):
    if kpts is None:
        return frame
    cv2.putText(frame, f"Trunk:{trunk_angle}°", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Knee:{knee_angle}°", (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    cv2.putText(frame, "LAST TAKEOFF STEP", (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# 新增：杆上重心、过杆效率
def calc_bar_center(kpts):
    if kpts is None:
        return -1.0
    sh_c = (kpts[5] + kpts[6]) / 2
    hip_c = (kpts[11] + kpts[12]) / 2
    return sh_c[1] * 0.3 + hip_c[1] * 0.7

def calc_over_bar_efficiency(center_y, bar_y):
    if center_y < 0 or bar_y < 0:
        return -1.0
    return center_y - bar_y