import cv2
import numpy as np
from scipy import signal
import math  # 用于计算躯干倾斜角

# ==============================================
# YOLO-Pose 人体关键点索引（固定不变）
# 5 = 左肩    6 = 右肩
# 11 = 左髋   12 = 右髋
# 13 = 左膝   14 = 右膝
# 15 = 左踝   16 = 右踝
# ==============================================

# ==============================================
# 函数功能：计算三个点之间的夹角（a-b-c，以 b 为中心点）
# 用途：计算膝盖角度、关节角度通用
# ==============================================
def calc_angle(a, b, c):
    # 计算向量 ba 和 bc
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    # 计算向量夹角公式
    dot = np.dot(ba, bc)
    mod_ba = np.linalg.norm(ba)
    mod_bc = np.linalg.norm(bc)

    # 防止除零
    if mod_ba == 0 or mod_bc == 0:
        return 0.0

    cos_angle = dot / (mod_ba * mod_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # 防止数值越界

    # 转为角度
    angle = math.degrees(math.acos(cos_angle))
    return angle

# ==============================================
# 平滑参数：控制关键点抖动
# MAX_HIST = 1 → 无平滑（追踪最灵敏，你现在用的）
# ==============================================
KP_HIST = []
MAX_HIST = 1

# ==============================================
# 函数功能：对关键点进行轻微平滑，减少抖动
# ==============================================
def smooth_kpts(kpts):
    global KP_HIST
    if kpts is None:
        KP_HIST.clear()
        return None

    KP_HIST.append(kpts.copy())
    if len(KP_HIST) > MAX_HIST:
        KP_HIST.pop(0)

    # 对最近几帧取平均，实现平滑
    mean_kpts = np.mean(np.array(KP_HIST), axis=0)
    return mean_kpts

# ==============================================
# 函数功能：计算人体重心（肩30% + 髋70%）
# 跳高动作分析最标准的重心计算方式
# ==============================================
def calc_center(kpts):
    # 取出左右髋
    left_hip  = kpts[11]
    right_hip = kpts[12]

    # 取出左右肩
    left_shoulder  = kpts[5]
    right_shoulder = kpts[6]

    # 关键点无效则返回 -1
    if left_hip[1] <= 1 or right_hip[1] <= 1:
        return -1

    # 髋中心点
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    if left_shoulder[1] <= 1 or right_shoulder[1] <= 1:
        return hip_center_y

    # 肩中心点
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # 重心 = 30%肩 + 70%髋
    center_y = shoulder_center_y * 0.3 + hip_center_y * 0.7
    return center_y

# ==============================================
# 函数功能：逐帧分析视频，提取关键点 + 重心
# 优化1：动态置信度 → 前面低/后面高
# ==============================================
def extract_all_frames(video_path, model, peak_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频")
        return None, None, 0, 0, 0, 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kpts = []
    center_ys = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ===================== 优化1：动态置信度 =====================
        # 助跑阶段：置信度 0.3 → 追踪更强
        # 起跳&过杆：置信度 0.4 → 更准，不飘
        if idx < peak_frame - 20:
            conf = 0.30
        else:
            conf = 0.40

        # 模型推理
        results = model(frame, verbose=False, conf=conf, imgsz=640)
        res = results[0]

        if res.keypoints and len(res.keypoints.xy) > 0:
            raw_kp = res.keypoints.xy[0].cpu().numpy()
            smooth_kp = smooth_kpts(raw_kp)
            all_kpts.append(smooth_kp)
            center_ys.append(calc_center(smooth_kp))
        else:
            KP_HIST.clear()
            all_kpts.append(None)
            center_ys.append(-1)

        idx += 1
        if idx % 50 == 0:
            print(f"分析进度: {idx}/{total}")

    cap.release()
    return all_kpts, center_ys, fps, width, height, total

# ==============================================
# 函数功能：平滑重心数据，让曲线更顺滑
# ==============================================
def smooth_center_ys(center_ys, window=5):
    ys = np.array(center_ys, dtype=float)
    ys[ys < 0] = np.nan
    nans = np.isnan(ys)

    if np.all(nans):
        return center_ys

    # 插值填充缺失值
    xs = np.arange(len(ys))
    ys[nans] = np.interp(xs[nans], xs[~nans], ys[~nans])

    # 中值滤波去抖动
    if window >= 3 and window % 2 == 1:
        ys = signal.medfilt(ys, kernel_size=window)

    ys = np.maximum(ys, 0)
    return ys.tolist()

# ==============================================
# 函数功能：把重心转为 0~100 高度曲线
# ==============================================
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

# ==============================================
# 函数功能：找到重心最高点（过杆帧）
# ==============================================
def find_peak_frame(norm_heights):
    if not norm_heights:
        return -1
    return int(np.argmax(norm_heights))

# ==============================================
# 【核心函数】计算：
# 1. 躯干倾斜角（已修复！）
# 2. 起跳腿膝角
# ==============================================
def compute_last_step_angles(kpts, takeoff_leg):
    if kpts is None:
        return None, None

    # ===================== 1. 膝角计算（正确）=====================
    # 根据起跳腿选择左/右侧关键点
    hip   = kpts[11] if takeoff_leg == "left" else kpts[12]
    knee  = kpts[13] if takeoff_leg == "left" else kpts[14]
    ankle = kpts[15] if takeoff_leg == "left" else kpts[16]

    # 计算 髋 → 膝 → 踝 的夹角
    knee_angle = calc_angle(hip, knee, ankle)

    # ===================== 2. 躯干倾斜角（已修复！）=====================
    # 取双肩、双髋
    sh_l, sh_r = kpts[5], kpts[6]
    hip_l, hip_r = kpts[11], kpts[12]

    # 计算肩中心点、髋中心点
    sx = (sh_l[0] + sh_r[0]) * 0.5
    sy = (sh_l[1] + sh_r[1]) * 0.5
    hx = (hip_l[0] + hip_r[0]) * 0.5
    hy = (hip_l[1] + hip_r[1]) * 0.5

    # 身体方向向量：髋 → 肩
    dx = sx - hx
    dy = sy - hy

    # 竖直向上的参考向量
    vx, vy = 0, -1

    # 向量夹角公式
    dot = dx * vx + dy * vy
    mod_body = math.sqrt(dx**2 + dy**2)
    cos_angle = dot / (mod_body * 1.0)
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    # 最终躯干角（与竖直线夹角）
    trunk_angle = math.degrees(math.acos(cos_angle))

    # 返回：躯干角度、膝角度
    return round(trunk_angle, 1), round(knee_angle, 1)

# ==============================================
# 函数功能：绘制姿态点 + 最后两步区间连线
# 优化4：连线区间扩大到 12 帧，覆盖完整最后两步
# ==============================================
def draw_pose(frame, kpts, config, frame_idx, target_frame, range_frame=12):
    if kpts is None:
        return frame

    # 1. 绘制所有关键点（绿色点）
    for x, y in kpts:
        if x > 5 and y > 5:
            cv2.circle(frame, (int(x), int(y)), 3, config.COLOR_POINT, -1)

    # 2. 只在最后两步区间内绘制连线
    if abs(frame_idx - target_frame) <= range_frame:
        # 肩中心、髋中心
        sh_l, sh_r = kpts[5], kpts[6]
        hip_l, hip_r = kpts[11], kpts[12]

        sh_c = (int((sh_l[0] + sh_r[0])/2), int((sh_l[1] + sh_r[1])/2))
        hip_c = (int((hip_l[0] + hip_r[0])/2), int((hip_l[1] + hip_r[1])/2))

        # 画躯干连线（黄色）
        cv2.line(frame, sh_c, hip_c, (0, 255, 255), 2)

        # 画起跳腿连线（髋→膝→踝）
        if config.TAKEOFF_LEG == "left":
            h, k, a = hip_l, kpts[13], kpts[15]
        else:
            h, k, a = hip_r, kpts[14], kpts[16]

        cv2.line(frame, (int(h[0]), int(h[1])), (int(k[0]), int(k[1])), (255, 100, 0), 2)
        cv2.line(frame, (int(k[0]), int(k[1])), (int(a[0]), int(a[1])), (255, 100, 0), 2)

    return frame

# ==============================================
# 函数功能：在最后一步绘制角度文字
# ==============================================
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