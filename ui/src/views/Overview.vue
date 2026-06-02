<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { api, fmtNum } from '../utils/api'
import * as echarts from 'echarts'

const router = useRouter()
const loading = ref(true)

const stats = ref([
  { label: '股票总数', value: '--', unit: '只', color: 'text-brand-600', trend: [] },
  { label: '交易日历', value: '--', unit: '天', color: 'text-brand-600', trend: [] },
  { label: '实际更新', value: '--', unit: '', color: 'text-slate-600', trend: [] },
  { label: '覆盖率', value: '--', unit: '%', color: 'text-success', trend: [] },
])

const quickActions = [
  { route: '/browser', label: '浏览数据', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' },
  { route: '/model',  label: '模型选股', icon: 'M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6' },
  { route: '/strategy', label: '策略调仓', icon: 'M3.75 6.75h16.5M3.75 12h10.5m-10.5 5.25h16.5m-6-6l3 3m0 0l-3 3m3-3H9.75' },
  { route: '/execution', label: '交易执行', icon: 'M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3' },
]

// 数据分布
const fieldStats = ref([])
let completenessChart = null

onMounted(async () => {
  const data = await api('/api/overview')
  if (data) {
    stats.value[0].value = data.stockCount || '--'
    stats.value[1].value = data.calendarDays || '--'
    stats.value[2].value = data.effectiveLastDate || data.lastUpdate || '--'
    stats.value[3].value = data.equityCount ? (data.equityCoverageAtLastDate * 100).toFixed(1) : '--'

    if (data.fieldStats) {
      fieldStats.value = data.fieldStats
    }
  }

  loading.value = false
  await nextTick()
  renderCompletenessChart()
})

onUnmounted(() => {
  completenessChart?.dispose()
})

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
    <!-- 统计卡片 (含骨架屏) -->
    <div v-if="loading" class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div v-for="i in 4" :key="i" class="bg-white rounded-xl border border-surface-3 p-4">
        <div class="skeleton h-3 w-20 mb-2"></div>
        <div class="flex items-baseline gap-1">
          <div class="skeleton h-7 w-16"></div>
          <div class="skeleton h-4 w-6"></div>
        </div>
      </div>
    </div>
    <div v-else class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div v-for="(s, i) in stats" :key="s.label"
           class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition group">
        <div class="flex items-start justify-between">
          <div>
            <div class="text-xs text-slate-500 mb-1">{{ s.label }}</div>
            <div class="flex items-baseline gap-1">
              <span :class="['text-2xl font-bold font-mono', s.color]">{{ s.value }}</span>
              <span class="text-sm text-slate-500">{{ s.unit }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 快捷入口 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">快捷操作</h2>
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <button v-for="a in quickActions" :key="a.route"
                @click="router.push(a.route)"
                :aria-label="a.label"
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
      <div v-if="loading" class="w-full h-[200px]"><div class="skeleton w-full h-full rounded-lg"></div></div>
      <div v-else id="completeness-chart" class="w-full h-[200px]"></div>
      <div v-if="!loading" class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-slate-600">
        <div class="rounded-lg bg-surface-1/60 border border-surface-3 px-4 py-3">
          <div class="text-xs text-slate-500 mb-1">实际更新日期</div>
          <div class="font-medium">{{ stats[2].value }}</div>
        </div>
        <div class="rounded-lg bg-surface-1/60 border border-surface-3 px-4 py-3">
          <div class="text-xs text-slate-500 mb-1">股票覆盖率</div>
          <div class="font-medium">{{ stats[3].value }}<span v-if="stats[3].value !== '--'">%</span></div>
        </div>
        <div class="rounded-lg bg-surface-1/60 border border-surface-3 px-4 py-3">
          <div class="text-xs text-slate-500 mb-1">说明</div>
          <div class="font-medium">日历推进不再等同于全市场已更新</div>
        </div>
      </div>
    </div>
  </div>
</template>
