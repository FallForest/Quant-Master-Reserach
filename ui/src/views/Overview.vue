<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { api, fmtNum } from '../utils/api'
import * as echarts from 'echarts'

const router = useRouter()

const stats = ref([
  { label: '股票总数', value: '--', unit: '只', color: 'text-brand-600', trend: [] },
  { label: '交易日历', value: '--', unit: '天', color: 'text-brand-600', trend: [] },
  { label: '最后更新', value: '--', unit: '', color: 'text-slate-600', trend: [] },
  { label: '数据完整度', value: '--', unit: '%', color: 'text-success', trend: [] },
])

const quickActions = [
  { route: '/pipeline', label: '更新数据', icon: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15' },
  { route: '/browser', label: '浏览数据', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' },
  { route: '/factor',  label: '因子分析', icon: 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z' },
]

// 数据分布
const fieldStats = ref([])
let completenessChart = null

onMounted(async () => {
  const data = await api('/api/overview')
  if (data) {
    stats.value[0].value = data.stockCount || '--'
    stats.value[1].value = data.calendarDays || '--'
    stats.value[2].value = data.lastUpdate || '--'
    stats.value[3].value = data.completeness || '--'

    if (data.fieldStats) {
      fieldStats.value = data.fieldStats
    }
  }

  // Generate demo sparkline data
  stats.value.forEach(s => {
    if (!s.trend.length) {
      s.trend = Array.from({ length: 14 }, () => 0.5 + Math.random() * 0.5)
    }
  })

  await nextTick()
  renderSparklines()
  renderCompletenessChart()
})

onUnmounted(() => {
  completenessChart?.dispose()
})

function renderSparklines() {
  stats.value.forEach((s, i) => {
    const el = document.getElementById(`sparkline-${i}`)
    if (!el || !s.trend.length) return
    const chart = echarts.init(el)
    chart.setOption({
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      xAxis: { type: 'category', show: false, data: s.trend.map((_, j) => j) },
      yAxis: { type: 'value', show: false, min: 'dataMin' },
      series: [{
        type: 'line',
        data: s.trend,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#3B82F6', width: 1.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59,130,246,0.25)' },
            { offset: 1, color: 'rgba(59,130,246,0)' },
          ]),
        },
      }],
    })
  })
}

function renderCompletenessChart() {
  const el = document.getElementById('completeness-chart')
  if (!el) return
  completenessChart = echarts.init(el)

  // Demo data completeness heatmap
  const fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adjclose', 'factor']
  const months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
  const data = []
  fields.forEach((f, i) => {
    months.forEach((m, j) => {
      const val = +(95 + Math.random() * 5).toFixed(1)
      data.push([j, i, val])
    })
  })

  completenessChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) {
        return `${fields[p.data[1]]} · ${months[p.data[0]]}<br/>完整度: <b>${p.data[2]}%</b>`
      },
    },
    grid: { left: 70, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: 'category',
      data: months,
      axisLabel: { color: '#94A3B8', fontSize: 10 },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'category',
      data: fields,
      axisLabel: { color: '#64748B', fontSize: 10, fontFamily: 'Fira Code' },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisTick: { show: false },
    },
    visualMap: {
      min: 90,
      max: 100,
      show: false,
      inRange: { color: ['#FEE2E2', '#FEF3C7', '#D1FAE5', '#10B981'] },
    },
    series: [{
      type: 'heatmap',
      data,
      label: {
        show: true,
        color: '#475569',
        fontSize: 9,
        fontFamily: 'Fira Code',
        formatter: p => `${p.data[2]}%`,
      },
      itemStyle: { borderColor: '#fff', borderWidth: 2, borderRadius: 3 },
    }],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">
    <!-- 统计卡片 -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div v-for="(s, i) in stats" :key="s.label"
           class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition group">
        <div class="flex items-start justify-between">
          <div>
            <div class="text-xs text-slate-400 mb-1">{{ s.label }}</div>
            <div class="flex items-baseline gap-1">
              <span :class="['text-2xl font-bold font-mono', s.color]">{{ s.value }}</span>
              <span class="text-sm text-slate-400">{{ s.unit }}</span>
            </div>
          </div>
          <div :id="`sparkline-${i}`" class="w-20 h-8 opacity-60 group-hover:opacity-100 transition"></div>
        </div>
      </div>
    </div>

    <!-- 快捷入口 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">快捷操作</h2>
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <button v-for="a in quickActions" :key="a.route"
                @click="router.push(a.route)"
                class="flex items-center gap-3 px-4 py-3 rounded-lg border border-surface-3
                       hover:border-brand-400 hover:bg-brand-50 transition cursor-pointer group">
          <svg class="w-5 h-5 text-brand-500 group-hover:text-brand-700 flex-shrink-0" fill="none"
               stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" :d="a.icon"/>
          </svg>
          <span class="text-sm font-medium text-slate-600 group-hover:text-brand-700">{{ a.label }}</span>
        </button>
      </div>
    </div>

    <!-- 数据完整度热力图 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">数据完整度</h2>
      <div id="completeness-chart" class="w-full h-[200px]"></div>
    </div>
  </div>
</template>
