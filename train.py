import sys
import os
import builtins
import importlib
import inspect
from pathlib import Path
import ultralytics.nn.modules as modules
from torch.utils.data import Dataset
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
import torchvision.transforms as T
from ultralytics.engine.trainer import BaseTrainer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# 确保在导入任何Ultralytics模块前执行monkey_patch
sys.path.insert(0, str(Path(__file__).parent))

# 你的自定义模块名（文件名不要写 .py）
custom_modules = [
    "NAFBlock",
    "ContrastAwareAttention",
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




if __name__ == '__main__':
    register_custom_modules()

    model = YOLO("traffic_signs.yaml")  # 你的 YAML 文件名
    model.train(
        data='data/data.yaml',
        imgsz=640,              # 图像尺寸
        batch=32,               # 最佳 batch size
        epochs=100,             # 初始设置为 100
        patience=50,            # 连续 20 个 epoch 无改进则停止
        device='0',             # 使用 GPU
        name='traffic_signs_yolo',
        project='runs/train',
        optimizer='AdamW',      # 推荐优化器
        lr0=0.001,              # 初始学习率
        lrf=0.001,               # 最终学习率 = lr0 * lrf
        momentum=0.9,
        weight_decay=0.0005,    # 权重衰减
        warmup_epochs=3,        # 学习率热身
        warmup_momentum=0.8,
        augment=True,
        warmup_bias_lr=0.1,
        cos_lr=True,            # 余弦学习率衰减
        amp=True,               # 自动混合精度
        pretrained=False,
        val=True,               # 每个 epoch 后验证
        workers=4,              # 数据加载线程
        dropout=0.1,            # 防止过拟合
        box=7.5,                # 边界框损失权重
        cls=0.5,                # 分类损失权重
        dfl=1.5                 # DFL 损失权重
    )