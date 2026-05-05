## Transformer
C:\Users\15728\Desktop\qlib\.venv\Scripts\python.exe -m qlib.cli.run C:\Users\15728\Desktop\qlib\examples\benchmarks\Transformer\workflow_config_transformer_Alpha360_2026_csi300_top20_ndrop1.yaml > C:\Users\15728\Desktop\qlib\transformer_alpha360_csi300_top20_ndrop1_gpu.log 2>&1

## DoubleEnsemble模型
C:\Users\15728\Desktop\qlib\.venv\Scripts\python.exe -m qlib.cli.run C:\Users\15728\Desktop\qlib\examples\benchmarks\DoubleEnsemble\workflow_config_doubleensemble_Alpha158_2026_local.yaml > C:\Users\15728\Desktop\qlib\reports\DoubleEnsemble\workflow_config_doubleensemble_Alpha158_2026_local.log 2>&1

## meta
C:\Users\15728\Desktop\qlib\.venv\Scripts\python.exe -m qlib.cli.run C:\Users\15728\Desktop\qlib\examples\benchmarks\MetaEnsemble\workflow_config_pretrained_metaensemble_Alpha158_2026_top20_ndrop1.yaml > C:\Users\15728\Desktop\qlib\reports\MetaEnsemble\workflow_config_pretrained_metaensemble_Alpha158_2026_top20_ndrop1.log 2>&1


看 2026-04-30 打分最高的前 20 只
.venv\Scripts\python.exe -c "import pandas as pd; p=r'C:\Users\15728\Desktop\qlib\mlruns\172014380832769268\987600268f824f809bf6f4de55ef3bee\artifacts\pred.pkl'; df=pd.read_pickle(p); x=df.xs(pd.Timestamp('2026-04-30'), level='datetime').sort_values('score', ascending=False).head(20); print(x.to_string())"

看 2026-04-30 回测里实际持仓股票
.venv\Scripts\python.exe -c "import pandas as pd; p=r'C:\Users\15728\Desktop\qlib\mlruns\172014380832769268\987600268f824f809bf6f4de55ef3bee\artifacts\portfolio_analysis\positions_normal_1day.pkl'; pos=pd.read_pickle(p); x=pos[pd.Timestamp('2026-04-30')].position; stocks=[k for k,v in x.items() if isinstance(v, dict)]; print('\n'.join(stocks))"

如果你想在 VS Code 里直接点着看，重点记住这个规则就行：

artifacts/pred.pkl = 模型排序
artifacts/portfolio_analysis/positions_normal_1day.pkl = 策略实际持仓
metrics/ = 各种指标数值
params/ = 这次运行用的参数
tags/ = mlflow 元信息