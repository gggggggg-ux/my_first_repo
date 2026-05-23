import cv2
import numpy as np
import math
import mediapipe as mp

# 人体骨架连接关系
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)
]

def smooth_center_ys(ys, window=5):
    smoothed = []
    pad = window // 2
    padded = [ys[0]]*pad + ys + [ys[-1]]*pad
    for i in range(len(ys)):
        segment = padded[i:i+window]
        valid = [y for y in segment if y > 0]
        smoothed.append(sum(valid)/len(valid) if valid else -1)
    return smoothed

def convert_to_math_coords(ys):
    valid = [y for y in ys if y > 0]
    if not valid:
        return []
    min_y, max_y = min(valid), max(valid)
    return [100*(max_y - y)/(max_y - min_y) if y>0 else -1 for y in ys]

def find_peak_frame(norm_heights):
    max_h = -1
    peak_idx = 0
    for i, h in enumerate(norm_heights):
        if h > max_h:
            max_h = h
            peak_idx = i
    return peak_idx

def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos = max(min(cos, 1.0), -1.0)
    return math.degrees(math.acos(cos))

# 绘制全身骨架连线 + 双腿、躯干角度标注
def draw_full_body_skeleton(frame, kpts, left_knee_angle, right_knee_angle, trunk_angle):
    if kpts is None:
        return
    # 绘制全身骨骼线条
    for p1, p2 in POSE_CONNECTIONS:
        pt1 = tuple(map(int, kpts[p1]))
        pt2 = tuple(map(int, kpts[p2]))
        cv2.line(frame, pt1, pt2, (200, 200, 0), 2)
    # 绘制关键点圆点
    for point in kpts:
        cv2.circle(frame, tuple(map(int, point)), 4, (0, 150, 255), -1)

    # 躯干角度标注
    sh_c = (kpts[11] + kpts[12]) / 2
    hip_c = (kpts[23] + kpts[24]) / 2
    cv2.putText(frame, f"Trunk:{trunk_angle:.1f}", (int(sh_c[0])+15, int(sh_c[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # 左腿角度
    cv2.putText(frame, f"L-Knee:{left_knee_angle:.1f}", (int(kpts[25][0])+15, int(kpts[25][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    # 右腿角度
    cv2.putText(frame, f"R-Knee:{right_knee_angle:.1f}", (int(kpts[26][0])+15, int(kpts[26][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# 姿态模型初始化 视频模式
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = PoseLandmarker.create_from_options(options)

def mediapipe_detect_pose(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(round(cv2.getTickCount() / cv2.getTickFrequency() * 1000))
    res = detector.detect_for_video(img, timestamp)

    if not res.pose_landmarks:
        return None

    kpts = np.zeros((33, 2), dtype=np.float32)
    for i, lm in enumerate(res.pose_landmarks[0]):
        kpts[i, 0] = lm.x * w
        kpts[i, 1] = lm.y * h
    return kpts

# 计算身体重心
def calc_body_center_mediapipe(kpts):
    if kpts is None:
        return -1.0
    sh_c = (kpts[11] + kpts[12]) / 2
    hip_c = (kpts[23] + kpts[24]) / 2
    return sh_c[1] * 0.3 + hip_c[1] * 0.7

# 同时计算双腿膝盖角度 + 躯干角度
def compute_both_leg_angles(kpts):
    if kpts is None:
        return 0.0, 0.0, 0.0
    # 左腿
    l_hip, l_knee, l_ankle = kpts[23], kpts[25], kpts[27]
    left_knee = calc_angle(l_hip, l_knee, l_ankle)
    # 右腿
    r_hip, r_knee, r_ankle = kpts[24], kpts[26], kpts[28]
    right_knee = calc_angle(r_hip, r_knee, r_ankle)
    # 躯干
    sh_c = (kpts[11] + kpts[12]) / 2
    hip_c = (kpts[23] + kpts[24]) / 2
    dx = sh_c[0] - hip_c[0]
    dy = sh_c[1] - hip_c[1]
    mod_body = math.hypot(dx, dy)
    trunk_angle = 0.0
    if mod_body != 0:
        cos_angle = dy / mod_body
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        trunk_angle = math.degrees(math.acos(cos_angle))
    return round(left_knee,1), round(right_knee,1), round(trunk_angle,1)

# 带颜色过滤的横杆检测（只识别红/黄黑杆，过滤白色球门和建筑线）
def detect_bar(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. 定义颜色范围：红色 + 黄色（覆盖红杆、黄黑相间杆）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # 合并颜色掩码
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = mask_red1 + mask_red2 + mask_yellow

    # 2. 形态学处理，去除噪点
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    color_mask = cv2.erode(color_mask, kernel, iterations=1)

    # 3. 在颜色区域内做边缘检测和直线检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.bitwise_and(edges, edges, mask=color_mask)

    # 4. 只在颜色区域内找长而平的线，过滤白色球门框和建筑线
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 120, 
        minLineLength=150, maxLineGap=8
    )

    bar_pos = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 必须接近水平
            if abs(y1 - y2) > 15:
                continue
            # 只保留画面中下部，过滤建筑顶部
            if y1 < frame.shape[0] * 0.4:
                continue
            # 线长必须足够长
            if (x2 - x1) < 150:
                continue
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            bar_pos = (x1, y1, x2, y2)

    return bar_pos