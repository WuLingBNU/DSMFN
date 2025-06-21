import torch
import random
import numpy as np


def set_seed(seed):
    # 设置随机种子
    seed = 42

    # 设置 Python 自身的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的 CPU 随机种子
    torch.manual_seed(seed)

    # 设置 PyTorch 的 CUDA 随机种子（如果使用 GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 适用于多 GPU 设置

    # 设置 PyTorch 后端随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
