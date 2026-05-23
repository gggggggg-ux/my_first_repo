import cv2
import numpy as np
import os
from datetime import datetime
import jump_utils_v2 as utils

os.makedirs("./结果", exist_ok=True)

def main():
    video_path = "myjump.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频打开失败")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 时间戳命名，防止覆盖
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./结果/跳高分析_{time_str}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("=== 跳高动作全分析程序启动 ===")

    # 首轮采集重心数据
    print("第一次分析：定位过杆关键帧...")
    frame_list = []
    center_ys = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame.copy())
        kpts = utils.mediapipe_detect_pose(frame)
        cy = utils.calc_body_center_mediapipe(kpts)
        center_ys.append(cy)

    if not frame_list:
        print("无有效视频帧")
        return

    smooth_ys = utils.smooth_center_ys(center_ys)
    norm_ys = utils.convert_to_math_coords(smooth_ys)
    peak_idx = utils.find_peak_frame(norm_ys)
    print(f"过杆重心峰值帧序号：{peak_idx}")

    # 二轮绘制全身骨架、角度、横杆
    print("第二次分析：生成全身标注视频...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i, frame in enumerate(frame_list):
        # 绘制横杆
        utils.detect_bar(frame)
        # 姿态检测
        kpts = utils.mediapipe_detect_pose(frame)
        if kpts is not None:
            l_knee, r_knee, trunk = utils.compute_both_leg_angles(kpts)
            # 绘制全身骨架+双腿躯干标注
            utils.draw_full_body_skeleton(frame, kpts, l_knee, r_knee, trunk)
            # 峰值标记
            if i == peak_idx:
                cv2.putText(frame, "PEAK", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 分析完成！文件保存至：{output_path}")

if __name__ == "__main__":
    main()