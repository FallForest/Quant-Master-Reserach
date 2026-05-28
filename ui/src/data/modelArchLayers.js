/**
 * Static layer definitions for model architecture diagrams.
 * Each key matches a model id from /api/model-catalog.
 * @type {Record<string, Array<{name: string, type: 'input'|'hidden'|'output', w: number}>>}
 */
export const archLayers = {
  lightgbm: [
    { name: '输入特征', type: 'input', w: 120 },
    { name: '梯度提升迭代 ×500', type: 'hidden', w: 160 },
    { name: '叶节点预测', type: 'output', w: 100 },
  ],
  lstm: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'LSTM ×2', type: 'hidden', w: 130 },
    { name: '全连接层', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  transformer: [
    { name: '输入嵌入', type: 'input', w: 100 },
    { name: '位置编码', type: 'hidden', w: 90 },
    { name: 'Multi-Head Attention ×3', type: 'hidden', w: 200 },
    { name: 'Feed-Forward', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  tcn: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: '因果卷积 (d=1,2,4,8)', type: 'hidden', w: 180 },
    { name: '残差连接', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  double_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 140 },
    { name: '残差学习', type: 'hidden', w: 100 },
    { name: 'Stacking (上层)', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  linear: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '线性变换 + L2正则', type: 'hidden', w: 150 },
    { name: '输出', type: 'output', w: 80 },
  ],
  sfm: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'STFT 频域分解', type: 'hidden', w: 130 },
    { name: '频率分量选择', type: 'hidden', w: 120 },
    { name: '状态空间建模', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  add: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '特征提取器', type: 'hidden', w: 110 },
    { name: '域判别器 (对抗)', type: 'hidden', w: 140 },
    { name: '域不变表示', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  tree_cn_lstm_rl: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LightGBM 特征提取', type: 'hidden', w: 160 },
    { name: 'CNN-LSTM 时序建模', type: 'hidden', w: 160 },
    { name: 'RL 策略优化', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  double_ensemble_residual_cn_lstm: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'Double Ensemble', type: 'hidden', w: 130 },
    { name: '残差计算', type: 'hidden', w: 90 },
    { name: 'CN-LSTM 残差学习', type: 'hidden', w: 150 },
    { name: '融合输出', type: 'output', w: 90 },
  ],
  adaptive_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N', type: 'hidden', w: 110 },
    { name: '市场状态感知', type: 'hidden', w: 120 },
    { name: '自适应加权', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  meta_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N 预测', type: 'hidden', w: 140 },
    { name: '元模型学习组合', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  dynamic_meta_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N 预测', type: 'hidden', w: 140 },
    { name: '滑动窗口元学习', type: 'hidden', w: 140 },
    { name: '动态权重更新', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  low_turnover_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '换手惩罚优化', type: 'hidden', w: 130 },
    { name: 'Stacking (上层)', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  residual_ensemble_lgb: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '残差信号提取', type: 'hidden', w: 120 },
    { name: 'LGB 残差校正', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  multiseed_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'Seed₁ 模型', type: 'hidden', w: 100 },
    { name: 'Seed₂ 模型 ...', type: 'hidden', w: 110 },
    { name: 'Seedₙ 模型', type: 'hidden', w: 100 },
    { name: '聚合 (mean)', type: 'output', w: 100 },
  ],
  cost_aware_ensemble: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '成本感知残差', type: 'hidden', w: 120 },
    { name: '净收益优化 Stacking', type: 'hidden', w: 160 },
    { name: '输出', type: 'output', w: 80 },
  ],
  pretrained_signal: [
    { name: '预训练模型权重', type: 'input', w: 140 },
    { name: '前向推理', type: 'hidden', w: 90 },
    { name: '信号输出', type: 'output', w: 90 },
  ],
  adarnn: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'RNN 编码器', type: 'hidden', w: 110 },
    { name: '梯度反转层', type: 'hidden', w: 110 },
    { name: '域判别器', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  localformer: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: '位置编码', type: 'hidden', w: 90 },
    { name: '局部注意力 ×N', type: 'hidden', w: 130 },
    { name: '前馈网络', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  hist: [
    { name: '股票特征', type: 'input', w: 100 },
    { name: '股票-概念图', type: 'hidden', w: 110 },
    { name: '股票-指数图', type: 'hidden', w: 110 },
    { name: '层次注意力', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  krnn: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'CNN 截面编码', type: 'hidden', w: 130 },
    { name: 'RNN 时序编码', type: 'hidden', w: 130 },
    { name: '融合输出', type: 'output', w: 90 },
  ],
  igmtf: [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '生成式因子交互', type: 'hidden', w: 150 },
    { name: '多头注意力', type: 'hidden', w: 110 },
    { name: '因子贡献归因', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  sandwich: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'CNN 编码器', type: 'hidden', w: 110 },
    { name: 'RNN 编码器', type: 'hidden', w: 110 },
    { name: '对称融合', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  tcts: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '预测器 (Forecaster)', type: 'hidden', w: 150 },
    { name: '权重优化器', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  tra: [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'LSTM/Transformer 骨干', type: 'hidden', w: 180 },
    { name: '多状态路由 ×N', type: 'hidden', w: 130 },
    { name: '自适应路径选择', type: 'hidden', w: 140 },
    { name: '输出', type: 'output', w: 80 },
  ],
  general_ptnn: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'nn.Module 包装', type: 'hidden', w: 130 },
    { name: '自定义训练循环', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  regime_horizon_cost_ensemble: [
    { name: '多视界标签', type: 'input', w: 110 },
    { name: '状态检测器', type: 'hidden', w: 110 },
    { name: '多视界基模型', type: 'hidden', w: 130 },
    { name: '成本感知混合', type: 'hidden', w: 120 },
    { name: '风险控制输出', type: 'output', w: 120 },
  ],
  transcendence_hybrid: [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '排名集成', type: 'hidden', w: 100 },
    { name: '残差分支', type: 'hidden', w: 100 },
    { name: '深度分支', type: 'hidden', w: 100 },
    { name: '验证目标融合', type: 'output', w: 130 },
  ],
  transcendence_signal_ensemble: [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '基模型 (Train/Valid)', type: 'hidden', w: 160 },
    { name: '验证集超参选择', type: 'hidden', w: 140 },
    { name: '加权聚合', type: 'output', w: 100 },
  ],
  topk_metalabel: [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '截面排名/阈值', type: 'hidden', w: 130 },
    { name: '元标签生成', type: 'hidden', w: 110 },
    { name: 'LightGBM 训练', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  hflgb: [
    { name: '1min 高频特征', type: 'input', w: 130 },
    { name: '梯度提升迭代', type: 'hidden', w: 120 },
    { name: '信号指标计算', type: 'hidden', w: 120 },
    { name: '换手控制', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
}

/** Default layers for models not in archLayers. */
export function defaultLayers(modelName) {
  return [
    { name: '输入', type: 'input', w: 80 },
    { name: modelName + ' 层', type: 'hidden', w: 140 },
    { name: '输出', type: 'output', w: 80 },
  ]
}
