"""全局共享状态：数据目录实例、行情实例、模型服务。"""

from .datadir import DataDir
from .tdx_quote import TDXQuote
from .model_service import ModelService

# 由 main() 初始化
data: DataDir = None
tdx_quote: TDXQuote = None
model_service: ModelService = None
