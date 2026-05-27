"""启动 API 服务器。

用法:
    python run_server.py [--data_dir ~/.quant_master/quant_master_data/tdx_cn_data]

或:
    python -m server
"""
from server.__main__ import main

if __name__ == "__main__":
    main()
