import requests

url = "https://mirrors.tuna.tsinghua.edu.cn/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
save_path = "pose_landmarker_lite.task"

print("开始下载模型文件")
res = requests.get(url, timeout=20)
with open(save_path, "wb") as f:
    f.write(res.content)
print("下载完成")