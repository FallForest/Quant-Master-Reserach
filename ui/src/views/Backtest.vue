<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { api } from '../utils/api'
import { useToast } from '../utils/toast'
import * as echarts from 'echarts'

const toast = useToast()

// ---- 状态 ----
const running = ref(false)
const progress = ref(0)
const stepText = ref('')
const logs = ref([])
const results = ref(null)
const loading = ref(false)
const activeStep = ref(0) // 0=model, 1=strategy, 2=backtest
const modelOptions = ref([])
const modelStrategyDefaults = ref({})
const strategyOptions = ref([])

// ---- 模型配置 ----
const modelCfg = ref({
  type: 'lightgbm',
  handler: 'Alpha158',
  train_start: '2018-01-01',
  train_end: '2023-12-31',
  valid_start: '2024-01-01',
  valid_end: '2025-12-31',
})

const handlerOptions = [
  { id: 'Alpha158', label: 'Alpha158 (158因子)' },
  { id: 'Alpha360', label: 'Alpha360 (360因子)' },
  { id: 'Alpha158vwap', label: 'Alpha158 VWAP (158因子+VWAP)' },
  { id: 'Alpha360vwap', label: 'Alpha360 VWAP (360因子+VWAP)' },
  { id: 'Alpha158LiquidityState', label: 'Alpha158 流动性状态' },
  { id: 'TranscendenceAlpha', label: 'Transcendence (300+因子)' },
]

// ---- 策略配置 ----
const strategyCfg = ref({
  type: 'topk_dropout',
  topk: 30,
  n_drop: 5,
  risk_degree: 0.95,
})

// ---- 回测配置 (回测起始 = 验证集结束 + 1天) ----
const backtestCfg = ref({
  buy_cost: 0.0015,
  sell_cost: 0.0015,
  open_cost: 0.0005,
  min_cost: 5,
  impact_cost: 0.0,
  slippage: 0.0,
  trade_unit: 100,
  limit_threshold: 0.095,
  shift: 1,
  return_type: 'close',
  bench: 'SH000905',
  start_date: '2026-01-02',
  end_date: '2026-12-31',
  freq: 'day',
})

let pollTimer = null
let cumChart = null

// 选模型时自动填充推荐策略参数
function applyModelDefaults(modelId) {
  const defaults = modelStrategyDefaults.value[modelId]
  if (!defaults) return
  strategyCfg.value.topk = defaults.topk
  strategyCfg.value.n_drop = defaults.n_drop
  if (defaults.open_cost != null) backtestCfg.value.open_cost = defaults.open_cost
  if (defaults.close_cost != null) backtestCfg.value.sell_cost = defaults.close_cost
  if (defaults.limit_threshold != null) backtestCfg.value.limit_threshold = defaults.limit_threshold
}

watch(() => modelCfg.value.type, (newId) => {
  applyModelDefaults(newId)
})

const steps = [
  { label: '模型', icon: 'M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9' },
  { label: '策略', icon: 'M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z' },
  { label: '回测', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
]

onMounted(async () => {
  loading.value = true
  // 并行加载：模型目录 + 策略目录 + 上次回测结果
  const [catalog, stratData, lastResult] = await Promise.all([
    api('/api/model-catalog'),
    api('/api/strategies'),
    api('/api/backtest/results/last'),
  ])
  // 模型列表
  if (catalog?.models) {
    modelOptions.value = catalog.models.map(m => ({ id: m.id, label: m.name, category: m.category }))
    // 存储模型推荐策略参数
    for (const m of catalog.models) {
      if (m.strategyDefaults) {
        modelStrategyDefaults.value[m.id] = m.strategyDefaults
      }
    }
    // 初始加载时填充默认模型的策略参数
    applyModelDefaults(modelCfg.value.type)
  }
  // 策略列表 (选股/增强/执行均可用于 backtest)
  if (stratData?.strategies) {
    strategyOptions.value = stratData.strategies
      .filter(s => ['选股', '增强', '执行'].includes(s.category))
      .map(s => ({ id: s.id, label: s.name, desc: s.desc, params: s.params }))
  }
  if (lastResult?.results) {
    results.value = lastResult.results
    await nextTick()
    renderCharts()
  }
  loading.value = false
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  cumChart?.dispose()
})

function appendLog(level, msg) {
  logs.value.push({ level, msg })
}

async function startBacktest() {
  running.value = true
  progress.value = 0
  logs.value = []
  results.value = null
  appendLog('info', '正在启动回测...')
  appendLog('info', `模型: ${modelCfg.value.type} / ${modelCfg.value.handler}`)
  appendLog('info', `策略: ${strategyOptions.value.find(s => s.id === strategyCfg.value.type)?.label}`)

  // 映射前端字段名 → 后端期望的字段名 (topk → top_k)
  const stratPayload = { ...strategyCfg.value, top_k: strategyCfg.value.topk }
  delete stratPayload.topk

  const payload = {
    model: modelCfg.value,
    strategy: stratPayload,
    ...backtestCfg.value,
  }

  const resp = await api('/api/backtest/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (resp?.runId) {
    pollStatus(resp.runId)
  } else {
    simulateRun()
  }
}

function cancelBacktest() {
  if (pollTimer) clearInterval(pollTimer)
  running.value = false
  appendLog('warn', '用户取消')
}

function pollStatus(runId) {
  pollTimer = setInterval(async () => {
    const st = await api(`/api/backtest/status/${runId}`)
    if (!st) return
    if (st.logs) st.logs.forEach(l => appendLog(l.level, l.msg))
    if (st.progress != null) progress.value = st.progress
    if (st.step) stepText.value = st.step
    if (st.done) {
      clearInterval(pollTimer)
      progress.value = 100
      if (st.success) {
        appendLog('success', '回测完成!')
        toast.success('回测完成')
        const res = await api(`/api/backtest/results/${runId}`)
        if (res?.results) {
          results.value = res.results
          await nextTick()
          renderCharts()
        }
      } else {
        appendLog('error', '回测失败')
        toast.error('回测失败')
      }
      running.value = false
    }
  }, 2000)
}

function simulateRun() {
  const simSteps = [
    { pct: 8, step: '加载模型', msg: `初始化 ${modelCfg.value.type} ...` },
    { pct: 15, step: '准备数据', msg: `读取 ${modelCfg.value.handler} 因子数据...` },
    { pct: 30, step: '训练模型', msg: `训练集: ${modelCfg.value.train_start} ~ ${modelCfg.value.train_end}` },
    { pct: 40, step: '生成信号', msg: '模型预测 → 信号分数' },
    { pct: 55, step: '策略构建', msg: `加载 ${strategyOptions.value.find(s => s.id === strategyCfg.value.type)?.label} 策略` },
    { pct: 70, step: '模拟交易', msg: '撮合订单，计算成本' },
    { pct: 85, step: '计算指标', msg: '收益、风险、换手率' },
    { pct: 95, step: '生成报告', msg: '绘制图表和汇总' },
    { pct: 100, step: '完成', msg: '回测完成!' },
  ]
  let i = 0
  pollTimer = setInterval(async () => {
    if (i >= simSteps.length) {
      clearInterval(pollTimer)
      const res = await api(`/api/backtest/results/demo`)
      if (!res?.results) results.value = null
      running.value = false
      toast.success('回测完成')
      return
    }
    progress.value = simSteps[i].pct
    stepText.value = simSteps[i].step
    appendLog('info', `[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] ${simSteps[i].msg}`)
    i++
  }, 1500)
}

// ---- 图表 ----
function renderCharts() { renderCumChart() }

function renderCumChart() {
  const el = document.getElementById('backtest-cum-chart')
  if (!el || !results.value?.daily?.length) return
  cumChart?.dispose()
  cumChart = echarts.init(el)
  const d = results.value.daily
  const dates = d.map(r => r.date)
  cumChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const dt = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const c = p.seriesName === '策略' ? '#3B82F6' : '#94A3B8'
          return `<span style="color:${c}">${p.seriesName}: ${p.value >= 0 ? '+' : ''}${p.value}%</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${dt}</div>${lines.join('<br>')}`
      },
    },
    legend: { data: ['策略', '基准'], top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 11 } },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: dates,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: '累计收益 %',
      nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code', formatter: v => `${v}%` },
    },
    series: [
      {
        name: '策略', type: 'line', data: d.map(r => r.cumReturn), smooth: true,
        lineStyle: { color: '#3B82F6', width: 2 }, itemStyle: { color: '#3B82F6' }, symbol: 'none',
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(59,130,246,0.15)' }, { offset: 1, color: 'rgba(59,130,246,0)' },
        ]) },
      },
      {
        name: '基准', type: 'line', data: d.map(r => r.benchCumReturn), smooth: true,
        lineStyle: { color: '#94A3B8', width: 1.5, type: 'dashed' }, itemStyle: { color: '#94A3B8' }, symbol: 'none',
      },
    ],
  })
}

const logColor = { info: 'text-brand-200', warn: 'text-amber-400', error: 'text-red-400', success: 'text-emerald-400' }
</script>

<template>
  <div class="p-4 sm:p-6 space-y-6 animate-slide-in">

    <!-- Pipeline 步骤条 -->
    <div class="bg-white rounded-xl border border-surface-3 p-4">
      <div class="flex items-center gap-2 mb-4">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-800">策略回测</h2>
        <span class="text-xs text-slate-400 ml-1">Model → Signal → Strategy → Backtest</span>
      </div>

      <!-- 步骤切换 -->
      <div class="flex items-center gap-1 mb-5">
        <template v-for="(s, i) in steps" :key="i">
          <button @click="activeStep = i"
            :class="['flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer transition-all',
                     activeStep === i ? 'bg-brand-600 text-white shadow-sm' :
                     activeStep > i ? 'bg-brand-50 text-brand-600' : 'bg-surface-2 text-slate-400 hover:text-slate-600']">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="s.icon"/>
            </svg>
            <span>{{ i + 1 }}. {{ s.label }}</span>
          </button>
          <svg v-if="i < steps.length - 1" class="w-4 h-4 text-slate-300 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5"/>
          </svg>
        </template>
      </div>

      <!-- Step 0: 模型配置 -->
      <div v-show="activeStep === 0">
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label class="block text-xs text-slate-500 mb-1">模型架构</label>
            <select v-model="modelCfg.type" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
              <option v-for="m in modelOptions" :key="m.id" :value="m.id">{{ m.label }} ({{ m.category }})</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">因子处理器</label>
            <select v-model="modelCfg.handler" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
              <option v-for="h in handlerOptions" :key="h.id" :value="h.id">{{ h.label }}</option>
            </select>
          </div>
          <div class="sm:col-span-2 lg:col-span-1">
            <div class="text-xs text-slate-500 mb-1">训练信号</div>
            <div class="flex items-center gap-2 text-xs text-slate-400 bg-surface-2/50 rounded-lg px-3 py-2">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244"/>
              </svg>
              模型预测输出 → 策略信号输入
            </div>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">训练起始</label>
            <input v-model="modelCfg.train_start" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">训练结束</label>
            <input v-model="modelCfg.train_end" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">验证集截止</label>
            <input v-model="modelCfg.valid_end" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
          </div>
        </div>
      </div>

      <!-- Step 1: 策略配置 -->
      <div v-show="activeStep === 1">
        <div class="space-y-4">
          <!-- 策略选择 -->
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div v-for="s in strategyOptions" :key="s.id" @click="strategyCfg.type = s.id"
              :class="['rounded-xl border p-4 cursor-pointer transition-all duration-200',
                       strategyCfg.type === s.id ? 'border-brand-400 ring-2 ring-brand-400/30 bg-brand-50/30 shadow-sm' : 'border-surface-3 hover:border-brand-200 hover:shadow-sm']">
              <div class="flex items-center gap-2 mb-1">
                <svg class="w-4 h-4" :class="strategyCfg.type === s.id ? 'text-brand-600' : 'text-slate-400'" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" :d="strategyCfg.type === s.id ? 'M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z' : 'M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z'"/>
                </svg>
                <span class="text-sm font-semibold" :class="strategyCfg.type === s.id ? 'text-brand-700' : 'text-slate-700'">{{ s.label }}</span>
              </div>
              <p class="text-xs text-slate-500 leading-relaxed">{{ s.desc }}</p>
            </div>
          </div>
          <!-- 策略参数 -->
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div v-if="strategyCfg.type !== 'enhanced_indexing'">
              <label class="block text-xs text-slate-500 mb-1">Top K (持有数量)</label>
              <input v-model.number="strategyCfg.topk" type="number" min="5" max="200" step="5"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div v-if="strategyCfg.type === 'topk_dropout'">
              <label class="block text-xs text-slate-500 mb-1">Dropout N (每期换入/换出)</label>
              <input v-model.number="strategyCfg.n_drop" type="number" min="1" max="50" step="1"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">风险度 (仓位比例)</label>
              <input v-model.number="strategyCfg.risk_degree" type="number" min="0.1" max="1.0" step="0.05"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
          </div>
          <!-- 信号流向说明 -->
          <div class="flex items-center gap-3 bg-surface-2/50 rounded-lg px-4 py-2.5 text-xs text-slate-500">
            <svg class="w-4 h-4 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244"/>
            </svg>
            <span>
              <b class="text-slate-600">{{ modelOptions.find(m => m.id === modelCfg.type)?.label }}</b>
              <span class="mx-1.5">→</span> 预测信号 (Signal)
              <span class="mx-1.5">→</span>
              <b class="text-slate-600">{{ strategyOptions.find(s => s.id === strategyCfg.type)?.label }}</b>
              <span class="mx-1.5">→</span> 交易决策
            </span>
          </div>
        </div>
      </div>

      <!-- Step 2: 回测参数 -->
      <div v-show="activeStep === 2">
        <!-- 交易成本 -->
        <div class="mb-4">
          <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">交易成本</h4>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label class="block text-xs text-slate-500 mb-1">买入佣金</label>
              <input v-model.number="backtestCfg.buy_cost" type="number" step="0.0001" min="0" max="0.01"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">卖出佣金</label>
              <input v-model.number="backtestCfg.sell_cost" type="number" step="0.0001" min="0" max="0.01"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">开户成本</label>
              <input v-model.number="backtestCfg.open_cost" type="number" step="0.0001" min="0" max="0.01"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">最低佣金 (元)</label>
              <input v-model.number="backtestCfg.min_cost" type="number" step="1" min="0" max="100"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">冲击成本</label>
              <input v-model.number="backtestCfg.impact_cost" type="number" step="0.0001" min="0" max="0.01"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">滑点</label>
              <input v-model.number="backtestCfg.slippage" type="number" step="0.001" min="0" max="0.05"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
          </div>
        </div>
        <!-- 交易规则 -->
        <div class="mb-4">
          <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">交易规则</h4>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label class="block text-xs text-slate-500 mb-1">交易单位 (股)</label>
              <input v-model.number="backtestCfg.trade_unit" type="number" step="100" min="100" max="10000"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">涨跌停阈值</label>
              <input v-model.number="backtestCfg.limit_threshold" type="number" step="0.005" min="0.05" max="0.2"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">信号延迟 (日)</label>
              <input v-model.number="backtestCfg.shift" type="number" step="1" min="0" max="5"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">收益计算方式</label>
              <select v-model="backtestCfg.return_type" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
                <option value="close">收盘价</option>
                <option value="vwap">VWAP</option>
              </select>
            </div>
          </div>
        </div>
        <!-- 回测范围 -->
        <div>
          <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">回测范围</h4>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label class="block text-xs text-slate-500 mb-1">基准指数</label>
              <select v-model="backtestCfg.bench" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
                <option value="SH000300">沪深300</option>
                <option value="SH000905">中证500</option>
                <option value="SH000852">中证1000</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">回测频率</label>
              <select v-model="backtestCfg.freq" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
                <option value="day">日线</option>
                <option value="1min">1分钟</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">开始日期</label>
              <input v-model="backtestCfg.start_date" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
            </div>
            <div>
              <label class="block text-xs text-slate-500 mb-1">结束日期</label>
              <input v-model="backtestCfg.end_date" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
            </div>
          </div>
        </div>
      </div>

      <!-- 底部操作栏 -->
      <div class="mt-5 flex items-center gap-3">
        <button @click="startBacktest" :disabled="running"
          class="flex items-center gap-2 px-6 py-2.5 bg-brand-600 text-white font-semibold rounded-lg hover:bg-brand-700 transition cursor-pointer shadow-sm disabled:opacity-50 disabled:cursor-not-allowed">
          <svg :class="['w-5 h-5', running && 'animate-spin']" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          <span>{{ running ? '回测中...' : '运行回测' }}</span>
        </button>
        <div v-if="!running" class="text-xs text-slate-400 flex items-center gap-4">
          <span class="flex items-center gap-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M21 7.5l-9-5.25L3 7.5"/></svg>
            {{ modelOptions.find(m => m.id === modelCfg.type)?.label }}
          </span>
          <svg class="w-3 h-3 text-slate-300" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5"/></svg>
          <span class="flex items-center gap-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846"/></svg>
            {{ strategyOptions.find(s => s.id === strategyCfg.type)?.label }}
          </span>
        </div>
        <span v-if="running" class="text-xs text-slate-400">回测耗时取决于数据量和策略复杂度</span>
      </div>
    </div>

    <!-- 进度面板 -->
    <div v-if="running || logs.length" class="bg-white rounded-xl border border-surface-3 overflow-hidden">
      <div class="px-5 py-3 border-b border-surface-3 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <svg v-if="running" class="w-4 h-4 text-brand-500 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25"/>
            <path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75"/>
          </svg>
          <span class="text-sm font-medium text-slate-700">{{ stepText || '准备就绪' }}</span>
        </div>
        <button v-if="running" @click="cancelBacktest" class="text-xs text-slate-400 hover:text-danger transition cursor-pointer">取消</button>
      </div>
      <div class="h-1.5 bg-surface-2">
        <div class="h-full bg-brand-500 rounded-r transition-all duration-300" :style="{ width: progress + '%' }"></div>
      </div>
      <div class="log-scroll bg-brand-950 text-brand-200 font-mono text-xs p-4 h-40 overflow-y-auto">
        <div v-for="(l, i) in logs" :key="i" :class="logColor[l.level] || logColor.info">{{ l.msg }}</div>
      </div>
    </div>

    <!-- 回测结果 -->
    <template v-if="results">
      <!-- 风险指标卡片 -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">年化收益</div>
          <div :class="['text-xl font-bold font-mono', results.metrics.annualReturn >= 0 ? 'text-bull' : 'text-bear']">
            {{ results.metrics.annualReturn >= 0 ? '+' : '' }}{{ results.metrics.annualReturn }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">基准 {{ results.metrics.benchAnnualReturn }}%</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">超额收益</div>
          <div :class="['text-xl font-bold font-mono', results.metrics.excessReturn >= 0 ? 'text-bull' : 'text-bear']">
            {{ results.metrics.excessReturn >= 0 ? '+' : '' }}{{ results.metrics.excessReturn }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">年化超额</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">夏普比率</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ results.metrics.sharpe }}</div>
          <div class="text-[10px] text-slate-400 mt-0.5">收益/风险</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">最大回撤</div>
          <div class="text-xl font-bold font-mono text-danger">-{{ results.metrics.maxDrawdown }}%</div>
          <div class="text-[10px] text-slate-400 mt-0.5">波动率 {{ results.metrics.std }}%</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">信息比率</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ results.metrics.informationRatio }}</div>
          <div class="text-[10px] text-slate-400 mt-0.5">胜率 {{ results.metrics.winRate }}%</div>
        </div>
        <!-- IC 指标 (仅真实回测) -->
        <template v-if="results.metrics.ic != null">
          <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
            <div class="text-[11px] text-slate-500 mb-1">IC</div>
            <div class="text-xl font-bold font-mono text-brand-600">{{ results.metrics.ic }}</div>
            <div class="text-[10px] text-slate-400 mt-0.5">ICIR {{ results.metrics.icir }}</div>
          </div>
          <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
            <div class="text-[11px] text-slate-500 mb-1">Rank IC</div>
            <div class="text-xl font-bold font-mono text-brand-600">{{ results.metrics.rankIc }}</div>
            <div class="text-[10px] text-slate-400 mt-0.5">Rank ICIR {{ results.metrics.rankIcir }}</div>
          </div>
        </template>
      </div>

      <!-- 累计收益曲线 -->
      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">累计收益对比</h3>
        <div id="backtest-cum-chart" class="w-full h-[320px]"></div>
      </div>

      <!-- 持仓 + 交易日志 -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="bg-white rounded-xl border border-surface-3 overflow-hidden">
          <div class="px-5 py-3 border-b border-surface-3">
            <h3 class="text-sm font-semibold text-slate-600">最新持仓 (Top {{ results.positions?.length || 0 }})</h3>
          </div>
          <div class="overflow-x-auto max-h-[300px] overflow-y-auto">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-white z-10">
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="px-4 py-2">股票代码</th>
                  <th class="px-4 py-2 text-right">权重</th>
                  <th class="px-4 py-2 text-right">当日盈亏</th>
                </tr>
              </thead>
              <tbody class="text-slate-700">
                <tr v-for="p in results.positions" :key="p.symbol" class="border-b border-surface-2/60 hover:bg-brand-50/30 transition">
                  <td class="px-4 py-2 font-mono text-xs text-brand-600">{{ p.symbol }}</td>
                  <td class="px-4 py-2 text-right font-mono">{{ (p.weight * 100).toFixed(2) }}%</td>
                  <td :class="['px-4 py-2 text-right font-mono font-medium', p.pnl >= 0 ? 'text-bull' : 'text-bear']">
                    {{ p.pnl >= 0 ? '+' : '' }}{{ p.pnl }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="bg-white rounded-xl border border-surface-3 overflow-hidden">
          <div class="px-5 py-3 border-b border-surface-3">
            <h3 class="text-sm font-semibold text-slate-600">交易日志</h3>
          </div>
          <div class="overflow-x-auto max-h-[300px] overflow-y-auto">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-white z-10">
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="px-4 py-2">日期</th>
                  <th class="px-4 py-2">代码</th>
                  <th class="px-4 py-2">方向</th>
                  <th class="px-4 py-2 text-right">价格</th>
                  <th class="px-4 py-2 text-right">成交量</th>
                  <th class="px-4 py-2 text-right">成本</th>
                </tr>
              </thead>
              <tbody class="text-slate-700">
                <tr v-for="(t, i) in results.trades" :key="i" class="border-b border-surface-2/60 hover:bg-brand-50/30 transition">
                  <td class="px-4 py-2 font-mono text-xs whitespace-nowrap">{{ t.date }}</td>
                  <td class="px-4 py-2 font-mono text-xs text-brand-600">{{ t.symbol }}</td>
                  <td class="px-4 py-2">
                    <span :class="['text-xs font-semibold px-1.5 py-0.5 rounded', t.direction === 'buy' ? 'bg-red-50 text-red-600' : 'bg-emerald-50 text-emerald-600']">
                      {{ t.direction === 'buy' ? '买入' : '卖出' }}
                    </span>
                  </td>
                  <td class="px-4 py-2 text-right font-mono">{{ t.price.toFixed(2) }}</td>
                  <td class="px-4 py-2 text-right font-mono text-xs">{{ (t.volume / 100).toFixed(0) }}手</td>
                  <td class="px-4 py-2 text-right font-mono text-xs text-slate-400">{{ t.cost.toFixed(2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- 交易统计 -->
      <div class="bg-white rounded-xl border border-surface-3 p-5">
        <h3 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">交易统计</h3>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div class="text-center p-3 bg-surface-2/50 rounded-lg">
            <div class="text-[10px] text-slate-500 mb-1">平均换手率</div>
            <div class="text-lg font-bold font-mono text-brand-600">{{ results.metrics.avgTurnover }}%</div>
          </div>
          <div class="text-center p-3 bg-surface-2/50 rounded-lg">
            <div class="text-[10px] text-slate-500 mb-1">总交易成本</div>
            <div class="text-lg font-bold font-mono text-warn">{{ results.metrics.totalCost }}%</div>
          </div>
          <div class="text-center p-3 bg-surface-2/50 rounded-lg">
            <div class="text-[10px] text-slate-500 mb-1">波动率</div>
            <div class="text-lg font-bold font-mono text-slate-600">{{ results.metrics.std }}%</div>
          </div>
          <div class="text-center p-3 bg-surface-2/50 rounded-lg">
            <div class="text-[10px] text-slate-500 mb-1">胜率</div>
            <div class="text-lg font-bold font-mono text-success">{{ results.metrics.winRate }}%</div>
          </div>
        </div>
      </div>

      <!-- 下一步操作 -->
      <div class="flex items-center gap-3 flex-wrap">
        <router-link to="/attribution"
          class="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-brand-600 rounded-lg hover:bg-brand-700 transition cursor-pointer">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6zM13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z"/>
          </svg>
          收益归因分析
        </router-link>
        <router-link to="/portfolio"
          class="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition cursor-pointer">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6"/>
          </svg>
          查看持仓
        </router-link>
        <router-link to="/optimizer"
          class="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition cursor-pointer">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z"/>
          </svg>
          组合优化
        </router-link>
      </div>
    </template>
  </div>
</template>
