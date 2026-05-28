<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

const loading = ref(true)
const data = ref(null)
const groupBy = ref('time')

let monthlyChart = null
let sectorChart = null

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  loading.value = true
  data.value = await api('/api/attribution')
  loading.value = false
  await nextTick()
  renderAll()
})

onUnmounted(() => {
  monthlyChart?.dispose(); sectorChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() { monthlyChart?.resize(); sectorChart?.resize() }

function renderAll() {
  renderMonthlyChart()
  renderSectorChart()
}

function renderMonthlyChart() {
  const el = document.getElementById('attr-monthly-chart')
  if (!el || !data.value?.monthly?.length) return
  monthlyChart?.dispose()
  monthlyChart = echarts.init(el)
  const d = data.value.monthly

  monthlyChart.setOption({
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const dt = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const c = p.seriesName === '资产配置' ? '#3B82F6' : p.seriesName === '个股选择' ? '#10B981' : '#F59E0B'
          return `<span style="color:${c}">${p.seriesName}: ${p.value >= 0 ? '+' : ''}${p.value}%</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${dt}</div>${lines.join('<br>')}`
      },
    },
    legend: { data: ['资产配置', '个股选择', '交互效应'], top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 10 } },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: d.map(r => r.month),
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: '收益贡献 %', nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, formatter: v => `${v}%` },
    },
    series: [
      {
        name: '资产配置', type: 'bar', stack: 'attr', data: d.map(r => r.raa),
        itemStyle: { color: 'rgba(59,130,246,0.7)', borderRadius: [0, 0, 0, 0] },
        barMaxWidth: 24,
      },
      {
        name: '个股选择', type: 'bar', stack: 'attr', data: d.map(r => r.rss),
        itemStyle: { color: 'rgba(16,185,129,0.7)' },
      },
      {
        name: '交互效应', type: 'bar', stack: 'attr', data: d.map(r => r.rin),
        itemStyle: { color: 'rgba(245,158,11,0.7)', borderRadius: [2, 2, 0, 0] },
      },
      {
        name: '合计', type: 'line', data: d.map(r => r.total),
        lineStyle: { color: '#EF4444', width: 2 }, itemStyle: { color: '#EF4444' }, symbol: 'circle', symbolSize: 6,
      },
    ],
  })
}

function renderSectorChart() {
  const el = document.getElementById('attr-sector-chart')
  if (!el || !data.value?.bySector?.length) return
  sectorChart?.dispose()
  sectorChart = echarts.init(el)
  const d = data.value.bySector

  sectorChart.setOption({
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
    },
    legend: { data: ['资产配置', '个股选择', '交互效应'], top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 10 } },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: d.map(r => r.sector),
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: '收益贡献 %', nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, formatter: v => `${v}%` },
    },
    series: [
      { name: '资产配置', type: 'bar', stack: 'attr', data: d.map(r => r.raa), itemStyle: { color: 'rgba(59,130,246,0.7)' }, barMaxWidth: 36 },
      { name: '个股选择', type: 'bar', stack: 'attr', data: d.map(r => r.rss), itemStyle: { color: 'rgba(16,185,129,0.7)' } },
      { name: '交互效应', type: 'bar', stack: 'attr', data: d.map(r => r.rin), itemStyle: { color: 'rgba(245,158,11,0.7)', borderRadius: [2, 2, 0, 0] } },
    ],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 切换 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z"/>
          <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">收益归因</h2>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
        <button v-for="g in [{v:'time',l:'按时间'},{v:'sector',l:'按行业'}]" :key="g.v"
          @click="groupBy = g.v"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                   groupBy === g.v ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
          {{ g.l }}
        </button>
      </div>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="space-y-4">
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div v-for="i in 4" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
          <div class="skeleton h-3 w-16 mb-2"></div>
          <div class="skeleton h-6 w-20"></div>
        </div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4"><div class="skeleton w-full h-[300px] rounded-lg"></div></div>
    </div>

    <template v-else-if="data">
      <!-- 汇总指标 -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">资产配置效应</div>
          <div :class="['text-xl font-bold font-mono', data.summary.allocation >= 0 ? 'text-bull' : 'text-bear']">
            {{ data.summary.allocation >= 0 ? '+' : '' }}{{ data.summary.allocation }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">RAA</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">个股选择效应</div>
          <div :class="['text-xl font-bold font-mono', data.summary.selection >= 0 ? 'text-bull' : 'text-bear']">
            {{ data.summary.selection >= 0 ? '+' : '' }}{{ data.summary.selection }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">RSS</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">交互效应</div>
          <div :class="['text-xl font-bold font-mono', data.summary.interaction >= 0 ? 'text-bull' : 'text-bear']">
            {{ data.summary.interaction >= 0 ? '+' : '' }}{{ data.summary.interaction }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">RIN</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">总超额收益</div>
          <div :class="['text-xl font-bold font-mono', data.summary.excessReturn >= 0 ? 'text-bull' : 'text-bear']">
            {{ data.summary.excessReturn >= 0 ? '+' : '' }}{{ data.summary.excessReturn }}%
          </div>
          <div class="text-[10px] text-slate-400 mt-0.5">基准 {{ data.summary.benchReturn }}%</div>
        </div>
      </div>

      <!-- Brinson 归因说明 -->
      <div class="bg-brand-50/50 rounded-xl border border-brand-100 px-4 py-3 text-xs text-slate-500 leading-relaxed">
        <span class="font-semibold text-brand-600">Brinson 归因模型：</span>
        将超额收益分解为 <b>资产配置效应 (RAA)</b>（行业超配/低配贡献）、<b>个股选择效应 (RSS)</b>（行业内选股贡献）和 <b>交互效应 (RIN)</b>（配置与选择的交叉影响）。
      </div>

      <!-- 图表 -->
      <div v-if="groupBy === 'time'" class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-1">月度归因分解</h3>
        <p class="text-[10px] text-slate-500 mb-2">堆叠柱状图：每月三项效应贡献 + 合计折线</p>
        <div id="attr-monthly-chart" class="w-full h-[300px]"></div>
      </div>

      <div v-if="groupBy === 'sector'" class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-1">行业归因分解</h3>
        <p class="text-[10px] text-slate-500 mb-2">各行业对组合超额收益的三项效应贡献</p>
        <div id="attr-sector-chart" class="w-full h-[300px]"></div>
      </div>
    </template>
  </div>
</template>
