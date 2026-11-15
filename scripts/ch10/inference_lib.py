"""分数 → 过线标签的唯一实现:评测、推理服务、阈值重演共用。

全类都不过线时取最高分类兜底,绝不产出空标签——这条规则只此一处,
三处调用口径才不会漂。不依赖 torch,推理服务可轻量引用。
"""
import numpy as np


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """probs: (n, num_classes);返回同形状 0/1 矩阵。"""
    preds = (np.asarray(probs) >= threshold).astype(int)
    for r in range(len(preds)):
        if preds[r].sum() == 0:
            preds[r][int(probs[r].argmax())] = 1
    return preds
