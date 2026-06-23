import torch
import numpy as np

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    MixUp数据增强方法

    Args:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数
        use_cuda: 是否使用CUDA

    Returns:
        混合后的数据和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp损失函数

    Args:
        criterion: 损失函数
        pred: 模型预测
        y_a, y_b: 两个标签
        lam: 混合比例

    Returns:
        混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    随机生成边界框用于CutMix

    Args:
        size: 图像尺寸
        lam: 混合比例

    Returns:
        边界框坐标
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # 均匀分布生成中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """
    CutMix数据增强方法

    Args:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数
        use_cuda: 是否使用CUDA

    Returns:
        混合后的数据和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda以匹配实际像素比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    CutMix损失函数

    Args:
        criterion: 损失函数
        pred: 模型预测
        y_a, y_b: 两个标签
        lam: 混合比例

    Returns:
        混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)