import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import jump_config as cfg
import jump_utils_v2 as utils
import bar_utils as bu

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    ensure_dir(cfg.OUT_DIR)
    print("加载模型...")
    pose_model = YOLO(cfg.MODEL_NAME)
    bar_model = bu.load_bar_model(cfg.BAR_MODEL)

    print("第一次分析：找最高点...")
    temp_kpts, temp_y, fps, w, h, total = utils.extract_all_frames(cfg.VIDEO_PATH, pose_model, 9999)
    temp_y = utils.smooth_center_ys(temp_y)
    temp_norm = utils.convert_to_math_coords(temp_y)
    peak = utils.find_peak_frame(temp_norm)
    last = max(0, peak - cfg.LAST_STEP_OFFSET)
    print(f"最高点帧: {peak}, 最后一步: {last}")

    print("正式分析...")
    all_kpts, center_ys, fps, width, height, total_frames = utils.extract_all_frames(cfg.VIDEO_PATH, pose_model, peak)
    center_ys = utils.smooth_center_ys(center_ys)
    norm_heights = utils.convert_to_math_coords(center_ys)
    kpts_last = all_kpts[last] if last < len(all_kpts) else None
    trunk_angle, knee_angle = utils.compute_last_step_angles(kpts_last, cfg.TAKEOFF_LEG)
    if trunk_angle is None:
        trunk_angle = knee_angle = 0.0

    bar_y_list = [-1]*total_frames
    bar_center_list = [-1.0]*total_frames
    eff_list = [-1.0]*total_frames

    out_h = height + cfg.GRAPH_HEIGHT
    out_w = width
    graph_h = cfg.GRAPH_HEIGHT
    margin_l, margin_r = cfg.MARGIN_LEFT, cfg.MARGIN_RIGHT
    margin_t, margin_b = cfg.MARGIN_TOP, cfg.MARGIN_BOTTOM
    graph_w = out_w - margin_l - margin_r
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.OUT_DIR, f"jump_final_{timestamp}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    curve_canvas = np.zeros((graph_h, out_w, 3), dtype=np.uint8)

    def update_curve(frame_idx):
        if frame_idx == 0:
            return
        prev_h = norm_heights[frame_idx-1]
        curr_h = norm_heights[frame_idx]
        if prev_h < 0 or curr_h < 0:
            return
        prev_x = margin_l + int((frame_idx-1)/total_frames*graph_w)
        curr_x = margin_l + int(frame_idx/total_frames*graph_w)
        base_y = graph_h - margin_b - 40
        full_range = graph_h - margin_t - margin_b - 60
        prev_y = base_y - int(prev_h * full_range / 100)
        curr_y = base_y - int(curr_h * full_range / 100)
        cv2.line(curve_canvas, (prev_x, prev_y), (curr_x, curr_y), cfg.COLOR_CURVE, 2)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        kpts = all_kpts[frame_idx] if frame_idx < len(all_kpts) else None

        if kpts is not None:
            utils.draw_pose(frame, kpts, cfg, frame_idx, last)
            if frame_idx == last:
                utils.draw_last_step_overlay(frame, kpts, trunk_angle, knee_angle, cfg.TAKEOFF_LEG)

        bar_info, bar_kpts = bu.detect_bar_and_pose(bar_model, frame, cfg.BAR_CONF)
        frame = bu.draw_bar(frame, bar_info, cfg.COLOR_BAR)
        frame = bu.draw_bar_pose(frame, bar_kpts)

        if bar_info is not None:
            bar_y = bar_info["bar_y"]
            bar_y_list[frame_idx] = bar_y
            bar_center_y = utils.calc_bar_center(bar_kpts)
            bar_center_list[frame_idx] = bar_center_y
            eff = utils.calc_over_bar_efficiency(bar_center_y, bar_y)
            eff_list[frame_idx] = eff
            cv2.putText(frame, f"Bar Y: {bar_y:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.COLOR_BAR, 2)
            cv2.putText(frame, f"Bar Center: {bar_center_y:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Efficiency: {eff:.1f} (0=BEST)", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.COLOR_EFF, 2)

        cv2.putText(frame, f"Frame: {frame_idx}  Time: {frame_idx/fps:.2f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Knee: {knee_angle:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Trunk: {trunk_angle:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        update_curve(frame_idx)
        lower = np.zeros((graph_h, out_w, 3), dtype=np.uint8)
        lower = cv2.addWeighted(lower, 0, curve_canvas, 1, 0)
        cv2.line(lower, (margin_l, graph_h-margin_b-40), (out_w-margin_r, graph_h-margin_b-40), (255,255,255), 1)
        cv2.line(lower, (margin_l, margin_t), (margin_l, graph_h-margin_b-40), (255,255,255), 1)
        for i in range(0,101,20):
            y = graph_h - margin_b -40 - int(i*(graph_h-margin_t-margin_b-60)/100)
            cv2.putText(lower, f"{i}%", (5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        for t in range(int(total_frames/fps)+1):
            x = margin_l + int(t/(total_frames/fps)*graph_w)
            cv2.putText(lower, f"{t}s", (x-10, graph_h-margin_b-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        curr_h = norm_heights[frame_idx]
        if curr_h >= 0:
            cx = margin_l + int(frame_idx/total_frames*graph_w)
            cy = graph_h - margin_b -40 - int(curr_h*(graph_h-margin_t-margin_b-60)/100)
            cv2.circle(lower, (cx,cy), 5, cfg.COLOR_MARKER, -1)
            cv2.line(lower, (cx,margin_t), (cx,graph_h-margin_b-40), cfg.COLOR_MARKER, 1)
        canvas = np.zeros((out_h, out_w,3), dtype=np.uint8)
        canvas[0:height, 0:width] = frame
        canvas[height:height+graph_h, 0:out_w] = lower
        out.write(canvas)

    cap.release()
    out.release()
    print(f"完成！视频已保存到：{out_path}")

if __name__ == "__main__":
    main()