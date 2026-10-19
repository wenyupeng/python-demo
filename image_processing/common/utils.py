import numpy as np  # lib for calculation
import os  # lib for operational system
import torch  # lib for deep learning
import random

from sympy import true


def seed_everything(seed: int) -> None:
    """Configure common random-number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # This is most reliable when set before Python starts.
    os.environ["PYTHONHASHSEED"] = str(seed)


def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    # 定义转换函数：将数字部分转换为整数，非数字部分转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)