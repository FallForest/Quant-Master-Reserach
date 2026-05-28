<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

const loading = ref(true)
const data = ref(null)
const selectedModel = ref('all')

let groupChart = null
let heatmapChart = null
let autocorrChart = null
let turnoverChart = null
let icHistChart = null

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  await loadData()
})

onUnmounted(() => {
  disposeAll()
  window.removeEventListener('resize', handleResize)
})

function disposeAll() {
  groupChart?.dispose(); heatmapChart?.dispose(); autocorrChart?.dispose()
  turnoverChart?.dispose(); icHistChart?.dispose()
}

function handleResize() {
  groupChart?.resize(); heatmapChart?.resize(); autocorrChart?.resize()
  turnoverChart?.resize(); icHistChart?.resize()
}

watch(selectedModel, loadData)

async function loadData() {
  loading.value = true
  const resp = await api(`/api/model-performance?model=${selectedModel.value}`)
  data.value = resp
  loading.value = false
  await nextTick()
  renderAllCharts()
}

function renderAllCharts() {
  if (!data.value) return
  renderGroupChart()
  renderHeatmap()
  renderAutocorrChart()
  renderTurnoverChart()
  renderIcHistChart()
}

function renderGroupChart() {
  const el = document.getElementById('mp-group-chart')
  if (!el || !data.value?.groupReturns) return
  groupChart?.dispose()
  groupChart = echarts.init(el)
  const gr = data.value.groupReturns
  const keys = Object.keys(gr).sort()
  if (!keys.length) return
  const dates = gr[keys[0]].map(d => d.date)
  const colors = { Q1: '#EF4444', Q2: '#F59E0B', Q3: '#94A3B8', Q4: '#3B82F6', Q5: '#8B5CF6', '多空': '#10B981' }

  groupChart.setOption({
    tooltip: {
      trigger: 'axis', backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const dt = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const c = colors[p.seriesName] || '#94A3B8'
          return `<span style="color:${c}">${p.seriesName}: ${p.value >= 0 ? '+' : ''}${p.value}%</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${dt}</div>${lines.join('<br>')}`
      },
    },
    legend: { data: keys, top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 10 } },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: dates,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: '累计收益 %', nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, formatter: v => `${v}%` },
    },
    series: keys.map(k => ({
      name: k, type: 'line', data: gr[k].map(d => d.value), smooth: true,
      lineStyle: { color: colors[k], width: k === 'Q1' || k === 'Q5' || k === '多空' ? 2.5 : 1 },
      itemStyle: { color: colors[k] }, symbol: 'none',
      areaStyle: k === '多空' ? { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(16,185,129,0.12)' }, { offset: 1, color: 'rgba(16,185,129,0)' },
      ]) } : undefined,
    })),
  })
}

function renderHeatmap() {
  const el = document.getElementById('mp-heatmap-chart')
  if (!el || !data.value?.icMonthly) return
  heatmapChart?.dispose()
  heatmapChart = echarts.init(el)
  const icm = data.value.icMonthly
  const months = Object.keys(icm)
  const values = Object.values(icm)

  heatmapChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) { return `${p.name}<br/>IC: <b>${p.value >= 0 ? '+' : ''}${p.value.toFixed(4)}</b>` },
    },
    grid: { left: 50, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: 'category', data: months,
      axisLabel: { color: '#94A3B8', fontSize: 9, rotate: 45 },
      axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: { type: 'value', show: false },
    visualMap: {
      min: Math.min(...values), max: Math.max(...values),
      show: false, inRange: { color: ['#EF4444', '#F8FAFC', '#3B82F6'] },
    },
    series: [{
      type: 'bar', data: values.map((v, i) => ({
        value: v, itemStyle: { color: v >= 0 ? 'rgba(59,130,246,0.7)' : 'rgba(239,68,68,0.7)', borderRadius: [2, 2, 0, 0] },
      })),
      barMaxWidth: 20,
    }],
  })
}

function renderAutocorrChart() {
  const el = document.getElementById('mp-autocorr-chart')
  if (!el || !data.value?.autocorrelation?.length) return
  autocorrChart?.dispose()
  autocorrChart = echarts.init(el)
  const ac = data.value.autocorrelation

  autocorrChart.setOption({
    tooltip: {
      trigger: 'axis', backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
    },
    grid: { left: 50, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: 'category', data: ac.map(d => `Lag${d.lag}`),
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10 },
    },
    series: [{
      type: 'bar', data: ac.map(d => ({
        value: d.value, itemStyle: { color: d.value >= 0 ? 'rgba(59,130,246,0.6)' : 'rgba(239,68,68,0.6)', borderRadius: [2, 2, 0, 0] },
      })),
      barMaxWidth: 16,
    }],
  })
}

function renderTurnoverChart() {
  const el = document.getElementById('mp-turnover-chart')
  if (!el || !data.value?.turnover) return
  turnoverChart?.dispose()
  turnoverChart = echarts.init(el)
  const top = data.value.turnover.top || []
  const bot = data.value.turnover.bottom || []

  turnoverChart.setOption({
    tooltip: {
      trigger: 'axis', backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
    },
    legend: { data: ['Top 组', 'Bottom 组'], top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 10 } },
    grid: { left: 50, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: top.map(d => d.date),
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, formatter: v => `${(v * 100).toFixed(0)}%` },
    },
    series: [
      { name: 'Top 组', type: 'line', data: top.map(d => d.value), smooth: true, lineStyle: { color: '#EF4444', width: 2 }, itemStyle: { color: '#EF4444' }, symbol: 'none' },
      { name: 'Bottom 组', type: 'line', data: bot.map(d => d.value), smooth: true, lineStyle: { color: '#3B82F6', width: 2 }, itemStyle: { color: '#3B82F6' }, symbol: 'none' },
    ],
  })
}

function renderIcHistChart() {
  const el = document.getElementById('mp-ic-hist-chart')
  if (!el || !data.value?.icHistogram?.length) return
  icHistChart?.dispose()
  icHistChart = echarts.init(el)
  const hist = data.value.icHistogram

  icHistChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) { return `IC: ${p.name}<br/>频次: <b>${p.value}</b>` },
    },
    grid: { left: 40, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: 'category', data: hist.map(d => d.bucket.toFixed(2)),
      axisLabel: { color: '#94A3B8', fontSize: 8, rotate: 45 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10 },
    },
    series: [{
      type: 'bar', data: hist.map(d => ({
        value: d.count,
        itemStyle: { color: d.bucket >= 0 ? 'rgba(59,130,246,0.6)' : 'rgba(239,68,68,0.6)', borderRadius: [2, 2, 0, 0] },
      })),
    }],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 选择器 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">模型绩效分析</h2>
      </div>
      <div class="flex-1"></div>
      <select v-model="selectedModel"
        class="px-3 py-1.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer">
        <option value="all">全部模型</option>
        <option value="lightgbm">LightGBM</option>
        <option value="double_ensemble">Double Ensemble</option>
        <option value="transformer">Transformer</option>
        <option value="lstm">LSTM</option>
      </select>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="space-y-5">
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div v-for="i in 5" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
          <div class="skeleton h-3 w-16 mb-2"></div>
          <div class="skeleton h-6 w-20"></div>
        </div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4"><div class="skeleton w-full h-[300px] rounded-lg"></div></div>
    </div>

    <template v-else-if="data">
      <!-- 汇总指标 -->
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">IC 均值</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ data.summary.icMean }}</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">IC 标准差</div>
          <div class="text-xl font-bold font-mono text-warn">{{ data.summary.icStd }}</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">ICIR</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ data.summary.icir }}</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">Rank IC</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ data.summary.rankIC }}</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">IC > 0 占比</div>
          <div class="text-xl font-bold font-mono text-success">{{ data.summary.icPositive }}%</div>
        </div>
      </div>

      <!-- 分组累计收益 -->
      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-1">分组累计收益</h3>
        <p class="text-[10px] text-slate-500 mb-2">按预测值排序等分5组 + 多空组合</p>
        <div id="mp-group-chart" class="w-full h-[300px]"></div>
      </div>

      <!-- 月度 IC + 自相关 + 换手率 + IC分布 -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">月度 IC 走势</h3>
          <div id="mp-heatmap-chart" class="w-full h-[220px]"></div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">预测自相关</h3>
          <p class="text-[10px] text-slate-500 mb-2">预测值滞后 1~20 期自相关系数</p>
          <div id="mp-autocorr-chart" class="w-full h-[220px]"></div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-1">组合换手率</h3>
          <p class="text-[10px] text-slate-500 mb-2">Top 组 vs Bottom 组换手率</p>
          <div id="mp-turnover-chart" class="w-full h-[220px]"></div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">IC 分布直方图</h3>
          <div id="mp-ic-hist-chart" class="w-full h-[220px]"></div>
        </div>
      </div>
    </template>
  </div>
</template>
