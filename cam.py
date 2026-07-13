import cv2
import os
import sys
import sys
import os
import cv2
import builtins
import importlib
import inspect
from pathlib import Path
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
rootPath = '/home/uw/桌面/AI+Education'
sys.path.insert(0,rootPath)

from Modules.get_camera_image import CameraImage
#iadlite

# 确保在导入任何Ultralytics模块前执行monkey_patch
sys.path.insert(0, str(Path(__file__).parent))

# 你的自定义模块名（文件名不要写 .py）
custom_modules = [
    "NAFBlock",
    "ContrastAwareAttention",
    "RandomWeatherEnhancementModule",
]

def register_custom_modules():
    # 获取必要的全局命名空间
    main_globals = sys.modules['__main__'].__dict__
    modules_globals = sys.modules['ultralytics.nn.modules'].__dict__
    tasks_globals = sys.modules['ultralytics.nn.tasks'].__dict__
    # 双重注册：确保在多个位置都能找到自定义模块
    for module_name in custom_modules:
        try:
            # 动态导入模块
            module = importlib.import_module(f"my_block.{module_name}")
            for name, cls in inspect.getmembers(module, inspect.isclass):
                # 1. 注册到 Ultralytics 模块路径
                setattr(modules, name, cls)
                # 2. 注册到全局命名空间
                builtins.__dict__[name] = cls
                # 3. 注册到主模块
                main_globals[name] = cls
                # 4. 注册到 ultralytics.nn.modules 的全局空间
                modules_globals[name] = cls
                # 5. 注册到 ultralytics.nn.tasks 的全局空间
                tasks_globals[name] = cls
                print(f"✅ 成功注册模块 {name} 到所有命名空间")
        except Exception as e:
            print(f"❌ 导入失败: {module_name} -> {e}")

# 注册自定义模块
register_custom_modules()

# 加载训练好的模型
model_path = '/home/uw/桌面/视觉类项目案例/15.自研项目部署/TrafficSigns/traffic.pt'
model = YOLO(model_path)

video = 'hk' # 内置摄像头
# video = '/dev/video0' # 外置摄像头

images_dir = "/home/uw/桌面/AI+Education/Competition/traffic"

try:
    scan = CameraImage(video)
except Exception as e:
    scan = None
    print(e)
cnt =1
while scan is not None:
    ret, image = scan.get_cam_image()
    if ret:
        image = cv2.resize(image, (640, 480))  # 调整分辨率

        # 推理并画框
        results = model.predict(source=image, imgsz=640, conf=0.25, device="cpu", verbose=False)
        image_1 = results[0].plot()  # 获取画好框的图

        cv2.imshow('img', image_1)  # 显示的是推理结果图像

        tap_str = cv2.waitKey(10)
        if tap_str == ord('q'):
            cv2.destroyAllWindows()
            break
        elif tap_str == ord('s'):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(images_dir, f"org{cnt:03d}.png")
            save_path_1 = os.path.join(images_dir, f"res{cnt:03d}.png")
            cnt += 1
            cv2.imwrite(save_path, image)      # 保存原图
            cv2.imwrite(save_path_1, image_1)  # 保存预测图
            print(f"Image saved to: {save_path}")
