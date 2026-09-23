# 模型文件

将匹配同一训练流程的 BEATs checkpoint 和三个 `.pkl` 后端复制到本目录，
再在 `../infer.yaml` 中设置路径。权重文件不纳入版本控制。
仅加载自己训练或可信来源的 checkpoint/pickle；其反序列化可执行 Python 代码。
