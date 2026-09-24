"""按源格式/目标格式选择转换器的工厂接口。"""


class TransferFactory:
    """注册并选择 PyTorch→ONNX、ONNX→OM 两个转换阶段。"""

    def __init__(self):
        """创建阶段映射，延迟导入计算框架，使环境检查不依赖完整推理环境。"""
        self.converters = {
            ("pt", "onnx"): self._export_onnx,
            ("onnx", "om"): self._compile_om,
        }

    def create(self, source: str, target: str):
        """返回指定格式转换器；不支持的转换组合抛出 ValueError。"""
        try:
            return self.converters[(source.lower(), target.lower())]
        except KeyError as error:
            raise ValueError(f"不支持的转换：{source} → {target}") from error

    def convert(self, source: str, target: str, config: dict) -> dict:
        """执行选中的转换器并返回成功结果；异常原样传播给调用方。"""
        return self.create(source, target)(config)

    @staticmethod
    def _export_onnx(config: dict) -> dict:
        """加载 ONNX 导出模块并执行 PyTorch→ONNX 转换及验证。"""
        from .onnx_export import export_onnx

        return export_onnx(config)

    @staticmethod
    def _compile_om(config: dict) -> dict:
        """加载 ATC 模块并执行 ONNX→OM 转换。"""
        from .atc import compile_om

        return compile_om(config)
