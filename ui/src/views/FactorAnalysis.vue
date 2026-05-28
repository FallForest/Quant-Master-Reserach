<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

// ---- 状态 ----
const selectedFactor = ref('all')
const loading = ref(true)

const metrics = ref([
  { label: 'IC 均值', value: '--', desc: '信息系数均值', color: 'text-brand-600' },
  { label: 'IC 标准差', value: '--', desc: '信息系数波动', color: 'text-warn' },
  { label: 'ICIR', value: '--', desc: 'IC 信息比率', color: 'text-brand-600' },
  { label: 'Rank IC', value: '--', desc: '排序信息系数', color: 'text-brand-600' },
  { label: 'IC > 0 占比', value: '--', desc: '正IC比例', color: 'text-success' },
  { label: '年化收益 (Q5)', value: '--', desc: 'Top 组合年化', color: 'text-bull' },
])

// 图表实例
let icChart = null
let groupChart = null
let heatmapChart = null

// ---- 数据加载 ----
onMounted(async () => {
  window.addEventListener('resize', handleResize)
  loading.value = true
  const data = await api(`/api/factor/analysis?factor=${selectedFactor.value}`)
  if (data) {
    applyData(data)
  } else {
    applyData(generateDemoData())
  }
  loading.value = false
  await nextTick()
  renderAllCharts()
})

onUnmounted(() => {
  icChart?.dispose()
  groupChart?.dispose()
  heatmapChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  icChart?.resize()
  groupChart?.resize()
  heatmapChart?.resize()
}

watch(selectedFactor, async () => {
  loading.value = true
  const data = await api(`/api/factor/analysis?factor=${selectedFactor.value}`)
  if (data) {
    applyData(data)
  } else {
    applyData(generateDemoData())
  }
  loading.value = false
  await nextTick()
  renderAllCharts()
})

// ---- 数据处理 ----
let icSeries = []
let groupReturns = {}
let corrMatrix = []
let corrFactors = []

function applyData(data) {
  if (data.metrics) {
    metrics.value[0].value = data.metrics.icMean ?? '--'
    metrics.value[1].value = data.metrics.icStd ?? '--'
    metrics.value[2].value = data.metrics.icir ?? '--'
    metrics.value[3].value = data.metrics.rankIC ?? '--'
    metrics.value[4].value = data.metrics.icPositive != null ? `${data.metrics.icPositive}%` : '--'
    metrics.value[5].value = data.metrics.annualReturnQ5 != null ? `${data.metrics.annualReturnQ5}%` : '--'
  }
  icSeries = data.icSeries || []
  groupReturns = data.groupReturns || {}
  corrMatrix = data.corrMatrix || []
  corrFactors = data.corrFactors || []
}

function icQuality(val) {
  if (val > 0.05) return { text: '优秀', cls: 'text-success bg-success/10' }
  if (val > 0.03) return { text: '良好', cls: 'text-brand-600 bg-brand-50' }
  if (val > 0) return { text: '一般', cls: 'text-warn bg-warn/10' }
  return { text: '弱', cls: 'text-danger bg-danger/10' }
}

function generateDemoData() {
  const days = []
  const start = new Date('2025-01-02')
  const end = new Date('2025-12-31')
  const cur = new Date(start)
  while (cur <= end) {
    const d = cur.getDay()
    if (d !== 0 && d !== 6) days.push(cur.toISOString().slice(0, 10))
    cur.setDate(cur.getDate() + 1)
  }

  const icData = days.map(date => {
    const ic = +((Math.random() - 0.42) * 0.15).toFixed(4)
    return { date, ic }
  })

  const icMean = +(icData.reduce((s, d) => s + d.ic, 0) / icData.length).toFixed(4)
  const icStd = +(Math.sqrt(icData.reduce((s, d) => s + (d.ic - icMean) ** 2, 0) / icData.length)).toFixed(4)
  const icir = icStd > 0 ? +(icMean / icStd).toFixed(4) : 0
  const positive = +(icData.filter(d => d.ic > 0).length / icData.length * 100).toFixed(1)

  const qDates = ['2025-01-02', '2025-04-01', '2025-07-01', '2025-10-01']
  const groups = {}
  for (let q = 1; q <= 5; q++) {
    groups[`Q${q}`] = qDates.map(date => ({
      date,
      value: +((6 - q) * 0.008 + (Math.random() - 0.5) * 0.04).toFixed(4),
    }))
  }

  const factors = ['Alpha158', 'Alpha360', 'Momentum', 'Value', 'Volatility', 'Size']
  const matrix = factors.map(() =>
    factors.map(() => +((Math.random() - 0.3) * 0.8).toFixed(2))
  )
  factors.forEach((_, i) => { matrix[i][i] = 1 })

  return {
    metrics: {
      icMean: icMean,
      icStd: icStd,
      icir: icir,
      rankIC: +(icMean * 1.1).toFixed(4),
      icPositive: positive,
      annualReturnQ5: +(18.5 + Math.random() * 10).toFixed(2),
    },
    icSeries: icData,
    groupReturns: groups,
    corrMatrix: matrix,
    corrFactors: factors,
  }
}

// ---- 图表渲染 ----
function renderAllCharts() {
  renderICChart()
  renderGroupChart()
  renderHeatmap()
}

function renderICChart() {
  const el = document.getElementById('ic-trend-chart')
  if (!el) return
  icChart?.dispose()
  icChart = echarts.init(el)

  const dates = icSeries.map(d => d.date)
  const ics = icSeries.map(d => d.ic)
  const icMean = ics.reduce((s, v) => s + v, 0) / ics.length

  // Rolling 20-day MA
  const ma20 = []
  let sum = 0
  const q = []
  for (const v of ics) {
    sum += v
    q.push(v)
    if (q.length > 20) sum -= q.shift()
    ma20.push(q.length >= 20 ? +(sum / 20).toFixed(4) : null)
  }

  icChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const d = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const color = p.value >= 0 ? '#EF4444' : '#10B981'
          return `<span style="color:${color}">${p.seriesName}: ${p.value?.toFixed(4)}</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${d}</div>${lines.join('<br>')}`
      },
    },
    grid: { left: 50, right: 20, top: 30, bottom: 30 },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' },
    },
    series: [
      {
        name: 'IC',
        type: 'bar',
        data: ics.map(v => ({
          value: v,
          itemStyle: { color: v >= 0 ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.7)' },
        })),
        barMaxWidth: 4,
      },
      {
        name: 'IC MA20',
        type: 'line',
        data: ma20,
        smooth: true,
        lineStyle: { color: '#F59E0B', width: 2 },
        itemStyle: { color: '#F59E0B' },
        symbol: 'none',
      },
      {
        name: 'IC 均值',
        type: 'line',
        data: dates.map(() => icMean),
        lineStyle: { color: '#3B82F6', width: 1, type: 'dashed' },
        itemStyle: { color: '#3B82F6' },
        symbol: 'none',
      },
    ],
  })
}

function renderGroupChart() {
  const el = document.getElementById('group-returns-chart')
  if (!el) return
  groupChart?.dispose()
  groupChart = echarts.init(el)

  const quantiles = Object.keys(groupReturns).sort()
  if (!quantiles.length) return

  const dates = groupReturns[quantiles[0]].map(d => d.date)
  const colors = ['#EF4444', '#F59E0B', '#94A3B8', '#3B82F6', '#8B5CF6']

  // Compute cumulative returns
  const series = quantiles.map((q, i) => {
    let cum = 1
    const cumData = groupReturns[q].map(d => {
      cum *= (1 + d.value)
      return +((cum - 1) * 100).toFixed(2)
    })
    return {
      name: q,
      type: 'line',
      data: cumData,
      smooth: true,
      lineStyle: { color: colors[i], width: i === 0 || i === quantiles.length - 1 ? 2.5 : 1 },
      itemStyle: { color: colors[i] },
      symbol: 'none',
      areaStyle: i === quantiles.length - 1
        ? { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(139,92,246,0.15)' },
            { offset: 1, color: 'rgba(139,92,246,0)' },
          ]) }
        : undefined,
    }
  })

  groupChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const d = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const color = p.seriesName === 'Q5' ? '#8B5CF6' : p.seriesName === 'Q1' ? '#EF4444' : '#94A3B8'
          return `<span style="color:${color}">${p.seriesName}: ${p.value >= 0 ? '+' : ''}${p.value}%</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${d}</div>${lines.join('<br>')}`
      },
    },
    legend: {
      data: quantiles,
      top: 4,
      right: 10,
      textStyle: { color: '#64748B', fontSize: 11, fontFamily: 'Fira Code' },
    },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '累计收益 %',
      nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: {
        color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code',
        formatter: v => `${v}%`,
      },
    },
    series,
  })
}

function renderHeatmap() {
  const el = document.getElementById('factor-corr-chart')
  if (!el) return
  heatmapChart?.dispose()
  heatmapChart = echarts.init(el)

  if (!corrMatrix.length) return

  const data = []
  corrMatrix.forEach((row, i) => {
    row.forEach((val, j) => {
      data.push([j, i, val])
    })
  })

  const absMax = Math.max(...corrMatrix.flat().map(Math.abs))

  heatmapChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) {
        const v = p.data[2]
        return `${corrFactors[p.data[1]]} × ${corrFactors[p.data[0]]}<br/><b>${v >= 0 ? '+' : ''}${v.toFixed(3)}</b>`
      },
    },
    grid: { left: 80, right: 40, top: 10, bottom: 60 },
    xAxis: {
      type: 'category',
      data: corrFactors,
      axisLabel: { color: '#64748B', fontSize: 10, rotate: 30, fontFamily: 'Fira Code' },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'category',
      data: corrFactors,
      axisLabel: { color: '#64748B', fontSize: 10, fontFamily: 'Fira Code' },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisTick: { show: false },
    },
    visualMap: {
      min: -absMax,
      max: absMax,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: {
        color: ['#10B981', '#F8FAFC', '#EF4444'],
      },
      textStyle: { color: '#64748B', fontSize: 10 },
    },
    series: [{
      type: 'heatmap',
      data,
      label: {
        show: true,
        color: '#1E293B',
        fontSize: 10,
        fontFamily: 'Fira Code',
        formatter: p => p.data[2].toFixed(2),
      },
      itemStyle: { borderColor: '#fff', borderWidth: 2, borderRadius: 3 },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' },
      },
    }],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">
    <!-- 工具栏 -->
    <div class="flex flex-wrap items-center gap-3">
      <h2 class="text-base font-semibold text-slate-700">因子分析</h2>
      <div class="flex-1"></div>
      <select v-model="selectedFactor"
        class="px-3 py-1.5 text-sm rounded-lg border border-surface-3 bg-white
               focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer">
        <option value="all">全部因子</option>
        <option value="alpha158">Alpha158</option>
        <option value="alpha360">Alpha360</option>
        <option value="momentum">动量因子</option>
        <option value="value">价值因子</option>
        <option value="volatility">波动率因子</option>
      </select>
    </div>

    <!-- KPI 指标卡片 (含骨架屏) -->
    <div v-if="loading" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <div v-for="i in 6" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
        <div class="skeleton h-3 w-16 mb-2"></div>
        <div class="skeleton h-6 w-20 mb-1"></div>
        <div class="skeleton h-2.5 w-24"></div>
      </div>
    </div>
    <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <div v-for="m in metrics" :key="m.label"
           class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
        <div class="text-[11px] text-slate-500 mb-1">{{ m.label }}</div>
        <div :class="['text-xl font-bold font-mono', m.color]">{{ m.value }}</div>
        <div class="text-[10px] text-slate-500 mt-0.5">{{ m.desc }}</div>
      </div>
    </div>

    <!-- IC 时序图 -->
    <div class="bg-white rounded-xl border border-surface-3 p-4">
      <div v-if="loading" class="h-[260px] flex items-center justify-center">
        <div class="skeleton w-full h-full rounded-lg"></div>
      </div>
      <template v-else>
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <h3 class="text-sm font-semibold text-slate-600">IC 时序走势</h3>
            <span class="text-[10px] px-2 py-0.5 rounded-full"
                  :class="icQuality(metrics[0].value).cls">
              {{ icQuality(metrics[0].value).text }}
            </span>
          </div>
          <div class="flex items-center gap-3 text-[10px] text-slate-500">
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-2 rounded-sm" style="background:rgba(239,68,68,0.7)"></span> 正IC
            </span>
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-2 rounded-sm" style="background:rgba(16,185,129,0.7)"></span> 负IC
            </span>
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-0.5 rounded" style="background:#F59E0B"></span> MA20
            </span>
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-0.5 rounded border-t border-dashed" style="border-color:#3B82F6"></span> 均值
            </span>
          </div>
        </div>
        <div id="ic-trend-chart" class="w-full h-[260px]"></div>
      </template>
    </div>

    <!-- 分组收益 + 因子相关性 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <!-- 分组累计收益 -->
      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">分组累计收益</h3>
        <p class="text-[10px] text-slate-500 mb-2">按因子值排序等分5组，Q1=最低，Q5=最高</p>
        <div v-if="loading" class="h-[280px]"><div class="skeleton w-full h-full rounded-lg"></div></div>
        <div v-else id="group-returns-chart" class="w-full h-[280px]"></div>
      </div>

      <!-- 因子相关性热力图 -->
      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">因子相关性矩阵</h3>
        <p class="text-[10px] text-slate-500 mb-2">Pearson 相关系数，越接近 ±1 相关性越强</p>
        <div v-if="loading" class="h-[280px]"><div class="skeleton w-full h-full rounded-lg"></div></div>
        <div v-else id="factor-corr-chart" class="w-full h-[280px]"></div>
      </div>
    </div>
  </div>
</template>
