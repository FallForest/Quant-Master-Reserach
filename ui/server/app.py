"""全局共享状态：数据目录实例、行情实例、Pipeline 计数器。"""
import threading

from .datadir import DataDir
from .tdx_quote import TDXQuote

# 由 main() 初始化
data: DataDir = None
tdx_quote: TDXQuote = None

# Pipeline 运行状态
pipeline_runs = {}
_pipeline_counter = 0
_pipeline_lock = threading.Lock()


def next_pipeline_counter():
    global _pipeline_counter
    with _pipeline_lock:
        _pipeline_counter += 1
        return _pipeline_counter
