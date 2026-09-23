"""ASD 推理结果及多分类器投票逻辑。"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClassifierResult:
    """单个后端结果：异常分数、独立阈值和阈值判断后的标签。"""

    name: str
    label: str
    score: float
    threshold: float


@dataclass(frozen=True)
class Result:
    """单个十秒窗口结果；最终 score 为异常票数，threshold 为所需票数 m。"""

    source: str
    window_index: int
    start_seconds: float
    valid_seconds: float
    classifiers: tuple[ClassifierResult, ...]
    label: str
    score: int
    threshold: int

    def to_dict(self) -> dict:
        """将本窗口及全部后端结果转换为可直接 JSON 序列化的字典。"""
        return asdict(self)


def decide(
    scores: dict[str, float],
    thresholds: dict[str, float],
    min_votes: int,
    normal_label: str,
    abnormal_label: str,
    source: str = "array",
    window_index: int = 0,
    start_seconds: float = 0.0,
    valid_seconds: float = 10.0,
) -> Result:
    """按分数不低于阈值判定后端，再按异常票数不低于 m 返回 Result。

    分数与阈值必须具有相同的非空后端集合，数值须有限且位于 [0, 1]。
    m 必须介于 1 与后端数之间；非法输入抛出 ValueError。
    """
    if not scores or scores.keys() != thresholds.keys():
        raise ValueError("分类器分数与阈值必须具有相同的非空后端集合")
    if type(min_votes) is not int or not 1 <= min_votes <= len(scores):
        raise ValueError("异常投票阈值 m 必须介于 1 与启用分类器数量之间")
    if not normal_label or not abnormal_label or normal_label == abnormal_label:
        raise ValueError("正常与异常标签必须为不同的非空字符串")
    results = []
    votes = 0
    for name, value in scores.items():
        score, threshold = float(value), float(thresholds[name])
        if not all(
            math.isfinite(item) and 0 <= item <= 1 for item in (score, threshold)
        ):
            raise ValueError(f"分类器 {name} 的分数或阈值不在 [0, 1] 内")
        abnormal = score >= threshold
        votes += int(abnormal)
        results.append(
            ClassifierResult(
                name, abnormal_label if abnormal else normal_label, score, threshold
            )
        )
    return Result(
        source,
        window_index,
        start_seconds,
        valid_seconds,
        tuple(results),
        abnormal_label if votes >= min_votes else normal_label,
        votes,
        min_votes,
    )
