<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

const loading = ref(true)
const models = ref([])
const categories = ref([])
const filterCategory = ref('all')
const expandedId = ref(null)
const selectedModel = ref(null)

let archChart = null

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  loading.value = true
  const data = await api('/api/model-catalog')
  models.value = data?.models || []
  categories.value = data?.categories || []
  loading.value = false
})

onUnmounted(() => {
  archChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  archChart?.resize()
}

const filtered = computed(() => {
  if (filterCategory.value === 'all') return models.value
  return models.value.filter(m => m.category === filterCategory.value)
})

function toggleExpand(m) {
  if (expandedId.value === m.id) {
    expandedId.value = null
    selectedModel.value = null
  } else {
    expandedId.value = m.id
    selectedModel.value = m
    nextTick(() => renderArchChart(m))
  }
}

function speedLabel(s) {
  return ['', '慢', '较慢', '中等', '较快', '快'][s] || '-'
}
function speedColor(s) {
  return ['', 'text-danger', 'text-warn', 'text-slate-500', 'text-brand-600', 'text-success'][s] || 'text-slate-400'
}

const categoryColors = {
  '树模型': 'bg-emerald-50 text-emerald-600',
  'RNN': 'bg-blue-50 text-blue-600',
  '注意力': 'bg-purple-50 text-purple-600',
  'CNN': 'bg-amber-50 text-amber-600',
  'DNN': 'bg-slate-100 text-slate-600',
  '集成': 'bg-rose-50 text-rose-600',
  '线性': 'bg-cyan-50 text-cyan-600',
  '频域': 'bg-indigo-50 text-indigo-600',
  'RL混合': 'bg-orange-50 text-orange-600',
  '高级集成': 'bg-pink-50 text-pink-600',
  '信号': 'bg-teal-50 text-teal-600',
  'Transcendence': 'bg-violet-50 text-violet-600',
}

const archLayers = {
  'lightgbm': [
    { name: '输入特征', type: 'input', w: 120 },
    { name: '梯度提升迭代 ×500', type: 'hidden', w: 160 },
    { name: '叶节点预测', type: 'output', w: 100 },
  ],
  'lstm': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'LSTM ×2', type: 'hidden', w: 130 },
    { name: '全连接层', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'transformer': [
    { name: '输入嵌入', type: 'input', w: 100 },
    { name: '位置编码', type: 'hidden', w: 90 },
    { name: 'Multi-Head Attention ×3', type: 'hidden', w: 200 },
    { name: 'Feed-Forward', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'tcn': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: '因果卷积 (d=1,2,4,8)', type: 'hidden', w: 180 },
    { name: '残差连接', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'double_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 140 },
    { name: '残差学习', type: 'hidden', w: 100 },
    { name: 'Stacking (上层)', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'linear': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '线性变换 + L2正则', type: 'hidden', w: 150 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'sfm': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'STFT 频域分解', type: 'hidden', w: 130 },
    { name: '频率分量选择', type: 'hidden', w: 120 },
    { name: '状态空间建模', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'add': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '特征提取器', type: 'hidden', w: 110 },
    { name: '域判别器 (对抗)', type: 'hidden', w: 140 },
    { name: '域不变表示', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'tree_cn_lstm_rl': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LightGBM 特征提取', type: 'hidden', w: 160 },
    { name: 'CNN-LSTM 时序建模', type: 'hidden', w: 160 },
    { name: 'RL 策略优化', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'double_ensemble_residual_cn_lstm': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'Double Ensemble', type: 'hidden', w: 130 },
    { name: '残差计算', type: 'hidden', w: 90 },
    { name: 'CN-LSTM 残差学习', type: 'hidden', w: 150 },
    { name: '融合输出', type: 'output', w: 90 },
  ],
  'adaptive_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N', type: 'hidden', w: 110 },
    { name: '市场状态感知', type: 'hidden', w: 120 },
    { name: '自适应加权', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'meta_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N 预测', type: 'hidden', w: 140 },
    { name: '元模型学习组合', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'dynamic_meta_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '基模型 ×N 预测', type: 'hidden', w: 140 },
    { name: '滑动窗口元学习', type: 'hidden', w: 140 },
    { name: '动态权重更新', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'low_turnover_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '换手惩罚优化', type: 'hidden', w: 130 },
    { name: 'Stacking (上层)', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'residual_ensemble_lgb': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '残差信号提取', type: 'hidden', w: 120 },
    { name: 'LGB 残差校正', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'multiseed_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'Seed₁ 模型', type: 'hidden', w: 100 },
    { name: 'Seed₂ 模型 ...', type: 'hidden', w: 110 },
    { name: 'Seedₙ 模型', type: 'hidden', w: 100 },
    { name: '聚合 (mean)', type: 'output', w: 100 },
  ],
  'cost_aware_ensemble': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'LGB ×N (底层)', type: 'hidden', w: 130 },
    { name: '成本感知残差', type: 'hidden', w: 120 },
    { name: '净收益优化 Stacking', type: 'hidden', w: 160 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'pretrained_signal': [
    { name: '预训练模型权重', type: 'input', w: 140 },
    { name: '前向推理', type: 'hidden', w: 90 },
    { name: '信号输出', type: 'output', w: 90 },
  ],
  'adarnn': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'RNN 编码器', type: 'hidden', w: 110 },
    { name: '梯度反转层', type: 'hidden', w: 110 },
    { name: '域判别器', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'localformer': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: '位置编码', type: 'hidden', w: 90 },
    { name: '局部注意力 ×N', type: 'hidden', w: 130 },
    { name: '前馈网络', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'hist': [
    { name: '股票特征', type: 'input', w: 100 },
    { name: '股票-概念图', type: 'hidden', w: 110 },
    { name: '股票-指数图', type: 'hidden', w: 110 },
    { name: '层次注意力', type: 'hidden', w: 110 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'krnn': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'CNN 截面编码', type: 'hidden', w: 130 },
    { name: 'RNN 时序编码', type: 'hidden', w: 130 },
    { name: '融合输出', type: 'output', w: 90 },
  ],
  'igmtf': [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '生成式因子交互', type: 'hidden', w: 150 },
    { name: '多头注意力', type: 'hidden', w: 110 },
    { name: '因子贡献归因', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'sandwich': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'CNN 编码器', type: 'hidden', w: 110 },
    { name: 'RNN 编码器', type: 'hidden', w: 110 },
    { name: '对称融合', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'tcts': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '预测器 (Forecaster)', type: 'hidden', w: 150 },
    { name: '权重优化器', type: 'hidden', w: 120 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'tra': [
    { name: '输入序列', type: 'input', w: 100 },
    { name: 'LSTM/Transformer 骨干', type: 'hidden', w: 180 },
    { name: '多状态路由 ×N', type: 'hidden', w: 130 },
    { name: '自适应路径选择', type: 'hidden', w: 140 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'general_ptnn': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: 'nn.Module 包装', type: 'hidden', w: 130 },
    { name: '自定义训练循环', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'regime_horizon_cost_ensemble': [
    { name: '多视界标签', type: 'input', w: 110 },
    { name: '状态检测器', type: 'hidden', w: 110 },
    { name: '多视界基模型', type: 'hidden', w: 130 },
    { name: '成本感知混合', type: 'hidden', w: 120 },
    { name: '风险控制输出', type: 'output', w: 120 },
  ],
  'transcendence_hybrid': [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '排名集成', type: 'hidden', w: 100 },
    { name: '残差分支', type: 'hidden', w: 100 },
    { name: '深度分支', type: 'hidden', w: 100 },
    { name: '验证目标融合', type: 'output', w: 130 },
  ],
  'transcendence_signal_ensemble': [
    { name: '因子输入', type: 'input', w: 100 },
    { name: '基模型 (Train/Valid)', type: 'hidden', w: 160 },
    { name: '验证集超参选择', type: 'hidden', w: 140 },
    { name: '加权聚合', type: 'output', w: 100 },
  ],
  'topk_metalabel': [
    { name: '输入特征', type: 'input', w: 100 },
    { name: '截面排名/阈值', type: 'hidden', w: 130 },
    { name: '元标签生成', type: 'hidden', w: 110 },
    { name: 'LightGBM 训练', type: 'hidden', w: 130 },
    { name: '输出', type: 'output', w: 80 },
  ],
  'hflgb': [
    { name: '1min 高频特征', type: 'input', w: 130 },
    { name: '梯度提升迭代', type: 'hidden', w: 120 },
    { name: '信号指标计算', type: 'hidden', w: 120 },
    { name: '换手控制', type: 'hidden', w: 100 },
    { name: '输出', type: 'output', w: 80 },
  ],
}

function renderArchChart(m) {
  const el = document.getElementById('model-arch-chart')
  if (!el) return
  archChart?.dispose()
  archChart = echarts.init(el)
  const layers = archLayers[m.id] || [
    { name: '输入', type: 'input', w: 80 },
    { name: m.name + ' 层', type: 'hidden', w: 140 },
    { name: '输出', type: 'output', w: 80 },
  ]
  const nodes = []
  const links = []
  const spacing = 100 / (layers.length + 1)
  layers.forEach((layer, i) => {
    const x = spacing * (i + 1)
    const color = layer.type === 'input' ? '#3B82F6' : layer.type === 'output' ? '#10B981' : '#8B5CF6'
    nodes.push({
      name: layer.name, x: x, y: 50,
      symbol: 'roundRect', symbolSize: [layer.w, 30],
      itemStyle: { color, borderRadius: 6 },
      label: { color: '#fff', fontSize: 10, fontFamily: 'Fira Code' },
    })
    if (i > 0) {
      links.push({ source: layers[i - 1].name, target: layer.name,
        lineStyle: { color: '#CBD5E1', width: 1.5 } })
    }
  })
  archChart.setOption({
    tooltip: { show: false },
    series: [{
      type: 'graph', layout: 'none', roam: false,
      data: nodes, links,
      lineStyle: { curveness: 0 },
      emphasis: { disabled: true },
    }],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 筛选 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">模型工坊</h2>
        <span class="text-xs text-slate-400 ml-1">{{ models.length }} 个模型</span>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5 overflow-x-auto">
        <button v-for="c in ['all', ...categories]" :key="c" @click="filterCategory = c"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200 whitespace-nowrap',
                   filterCategory === c ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
          {{ c === 'all' ? '全部' : c }}
        </button>
      </div>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      <div v-for="i in 6" :key="i" class="bg-white rounded-xl border border-surface-3 p-4">
        <div class="flex items-center gap-3 mb-3">
          <div class="skeleton h-5 w-5 rounded"></div>
          <div class="skeleton h-4 w-24"></div>
          <div class="skeleton h-5 w-12 rounded-full"></div>
        </div>
        <div class="skeleton h-3 w-full mb-1"></div>
        <div class="skeleton h-3 w-3/4"></div>
      </div>
    </div>

    <!-- 模型卡片网格 -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      <div v-for="m in filtered" :key="m.id"
           @click="toggleExpand(m)"
           :class="['bg-white rounded-xl border border-surface-3 overflow-hidden cursor-pointer transition-all duration-200',
                    expandedId === m.id ? 'ring-2 ring-brand-400 shadow-sm md:col-span-2 xl:col-span-3' : 'hover:shadow-sm hover:border-brand-200']">
        <!-- 卡片头部 -->
        <div class="p-4">
          <div class="flex items-center gap-2 mb-2">
            <svg class="w-5 h-5 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9"/>
            </svg>
            <span class="font-semibold text-sm text-slate-700">{{ m.name }}</span>
            <span :class="['text-[10px] px-2 py-0.5 rounded-full font-medium', categoryColors[m.category] || 'bg-surface-2 text-slate-500']">
              {{ m.category }}
            </span>
            <div class="flex-1"></div>
            <div class="flex items-center gap-1">
              <span class="text-[10px] text-slate-400">速度</span>
              <span :class="['text-xs font-semibold', speedColor(m.speed)]">{{ speedLabel(m.speed) }}</span>
            </div>
          </div>
          <p class="text-xs text-slate-500 leading-relaxed line-clamp-2">{{ m.desc }}</p>
          <!-- 复杂度条 -->
          <div class="mt-2.5 flex items-center gap-2">
            <span class="text-[10px] text-slate-400">复杂度</span>
            <div class="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
              <div class="h-full rounded-full bg-brand-400 transition-all duration-300"
                   :style="{ width: (m.complexity / 5 * 100) + '%' }"></div>
            </div>
            <span class="text-[10px] font-mono text-slate-400">{{ m.complexity }}/5</span>
          </div>
        </div>

        <!-- 展开详情 -->
        <div v-if="expandedId === m.id" class="px-4 pb-4 border-t border-surface-3">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
            <!-- 参数 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">超参数配置</h4>
              <div class="bg-surface-2/50 rounded-lg p-3 space-y-2">
                <div v-for="(p, k) in m.params" :key="k" class="flex items-center justify-between text-xs">
                  <span class="text-slate-500">{{ p.label }}</span>
                  <span class="text-slate-700 font-mono">
                    {{ p.default }}{{ p.min != null ? ` [${p.min}~${p.max}]` : '' }}
                  </span>
                </div>
              </div>
            </div>
            <!-- 架构图 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">模型架构</h4>
              <div id="model-arch-chart" class="w-full h-[120px] bg-surface-2/50 rounded-lg"></div>
            </div>
          </div>
          <div class="mt-3 flex items-center gap-2">
            <router-link :to="{ path: '/backtest', query: { model: m.id } }"
              class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-brand-600 rounded-lg hover:bg-brand-700 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              用此模型回测
            </router-link>
            <router-link to="/experiments"
              class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
              </svg>
              查看实验
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
