# 视频路径（你给的）
VIDEO_PATH = r"D:\桌面\跳高模型\git\myjump.mp4"

# 姿态模型
MODEL_NAME = 'yolov8s-pose.pt'
TAKEOFF_LEG = 'left'
LAST_STEP_OFFSET = 18

# 横杆模型（你给的）
BAR_MODEL = r"D:\桌面\跳高模型\跳高训练\runs\detect\train-2\weights\best.pt"
BAR_CONF = 0.6

# 输出结果目录（你要的：结果文件夹）
OUT_DIR = r"D:\桌面\跳高模型\结果"

# 可视化
GRAPH_HEIGHT = 200
MARGIN_LEFT = 60
MARGIN_RIGHT = 40
MARGIN_TOP = 20
MARGIN_BOTTOM = 40

# 颜色
COLOR_POINT = (0, 255, 0)
COLOR_CURVE = (0, 255, 0)
COLOR_MARKER = (0, 0, 255)
COLOR_BAR = (0, 255, 255)
COLOR_EFF = (0, 255, 0)