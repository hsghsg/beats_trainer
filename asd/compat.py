"""将历史模型的 pickle 模块路径映射到 ASD 内的源码副本。"""

import pickle
from importlib import import_module


class Unpickler(pickle.Unpickler):
    """支持 beats_trainer 与 BEATs 旧模块名的兼容反序列化器。"""

    def find_class(self, module: str, name: str):
        """将项目类解析为 vendor 副本，第三方类型保留标准解析；仅加载可信产物。"""
        if module.split(".")[0] in {"beats_trainer", "BEATs"}:
            module = f"{__package__}.vendor.{module}"
            return getattr(import_module(module), name)
        return super().find_class(module, name)


def load(file, **kwargs):
    """从二进制文件读取兼容 pickle 对象，供 torch 和后端加载调用。"""
    return Unpickler(file, **kwargs).load()
