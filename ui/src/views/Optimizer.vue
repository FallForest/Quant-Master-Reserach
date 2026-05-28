<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

const loading = ref(true)
const methods = ref([])
const comparison = ref([])
const sectors = ref([])
const selectedMethod = ref(null)
const activeTab = ref('config')

let weightChart = null
let frontierChart = null

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  loading.value = true
  const data = await api('/api/optimizer')
  methods.value = data?.methods || []
  comparison.value = data?.comparison || []
  sectors.value = data?.sectors || []
  if (methods.value.length) selectedMethod.value = methods.value[0]
  loading.value = false
  await nextTick()
  renderCharts()
})

onUnmounted(() => {
  weightChart?.dispose(); frontierChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() { weightChart?.resize(); frontierChart?.resize() }

watch(selectedMethod, () => { nextTick(renderCharts) })

function renderCharts() {
  renderWeightChart()
  renderFrontierChart()
}

function renderWeightChart() {
  const el = document.getElementById('opt-weight-chart')
  if (!el || !comparison.value.length) return
  weightChart?.dispose()
  weightChart = echarts.init(el)
  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B']

  weightChart.setOption({
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const dt = params[0]?.axisValue || ''
        const lines = params.map(p => {
          const c = colors[p.seriesIndex] || '#94A3B8'
          return `<span style="color:${c}">${p.seriesName}: ${(p.value * 100).toFixed(1)}%</span>`
        })
        return `<div style="font-weight:600;margin-bottom:4px">${dt}</div>${lines.join('<br>')}`
      },
    },
    legend: { data: comparison.value.map(c => c.method), top: 4, right: 10, textStyle: { color: '#64748B', fontSize: 10 } },
    grid: { left: 55, right: 20, top: 36, bottom: 30 },
    xAxis: {
      type: 'category', data: sectors.value,
      axisLabel: { color: '#94A3B8', fontSize: 10 }, axisLine: { lineStyle: { color: '#E2E8F0' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, formatter: v => `${(v * 100).toFixed(0)}%` },
    },
    series: comparison.value.map((c, i) => ({
      name: c.method, type: 'bar', data: c.weights,
      itemStyle: { color: colors[i], borderRadius: [2, 2, 0, 0] },
      barMaxWidth: 24,
    })),
  })
}

function renderFrontierChart() {
  const el = document.getElementById('opt-frontier-chart')
  if (!el || !comparison.value.length) return
  frontierChart?.dispose()
  frontierChart = echarts.init(el)
  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B']

  // Generate efficient frontier curve
  const frontier = []
  for (let r = 8; r <= 28; r += 0.5) {
    const risk = r + Math.sin(r * 0.3) * 2 + 3
    frontier.push([risk, r])
  }

  const scatterData = comparison.value.map((c, i) => ({
    value: [c.risk, c.return],
    name: c.method,
    itemStyle: { color: colors[i] },
  }))

  frontierChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) {
        if (p.data?.name) return `${p.data.name}<br/>收益: ${p.data.value[1]}%<br/>风险: ${p.data.value[0]}%`
        return `风险: ${p.value[0].toFixed(1)}%<br/>收益: ${p.value[1].toFixed(1)}%`
      },
    },
    grid: { left: 60, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'value', name: '风险 (%)', nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10 },
    },
    yAxis: {
      type: 'value', name: '收益 (%)', nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10 },
    },
    series: [
      {
        name: '有效前沿', type: 'line', data: frontier, smooth: true, symbol: 'none',
        lineStyle: { color: '#CBD5E1', width: 1.5, type: 'dashed' },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(59,130,246,0.06)' }, { offset: 1, color: 'rgba(59,130,246,0)' },
        ]) },
      },
      {
        name: '策略', type: 'scatter', data: scatterData, symbolSize: 14,
        label: { show: true, formatter: p => p.data.name, position: 'right', fontSize: 10, color: '#475569' },
      },
    ],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">组合优化</h2>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
        <button v-for="t in [{v:'config',l:'参数配置'},{v:'compare',l:'方法对比'}]" :key="t.v"
          @click="activeTab = t.v"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                   activeTab === t.v ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
          {{ t.l }}
        </button>
      </div>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="space-y-4">
      <div class="bg-white rounded-xl border border-surface-3 p-5">
        <div class="skeleton h-4 w-40 mb-4"></div>
        <div class="grid grid-cols-3 gap-3">
          <div v-for="i in 5" :key="i" class="skeleton h-10 rounded-lg"></div>
        </div>
      </div>
    </div>

    <template v-else>
      <!-- 参数配置 -->
      <div v-if="activeTab === 'config'" class="space-y-4">
        <!-- 方法选择 -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          <div v-for="m in methods" :key="m.id" @click="selectedMethod = m"
            :class="['bg-white rounded-xl border p-4 cursor-pointer transition-all duration-200',
                     selectedMethod?.id === m.id ? 'border-brand-400 ring-2 ring-brand-400/30 shadow-sm' : 'border-surface-3 hover:border-brand-200 hover:shadow-sm']">
            <div class="flex items-center gap-2 mb-1.5">
              <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z"/>
                <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z"/>
              </svg>
              <span class="text-sm font-semibold text-slate-700">{{ m.name }}</span>
            </div>
            <p class="text-xs text-slate-500 leading-relaxed">{{ m.desc }}</p>
          </div>
        </div>

        <!-- 选中方法的参数 -->
        <div v-if="selectedMethod" class="bg-white rounded-xl border border-surface-3 p-5">
          <h3 class="text-sm font-semibold text-slate-600 mb-4">{{ selectedMethod.name }} — 参数配置</h3>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div v-for="(p, k) in selectedMethod.params" :key="k">
              <label class="block text-xs text-slate-500 mb-1">{{ p.label }}</label>
              <input v-if="p.type === 'number'" type="number"
                :value="p.default" :min="p.min" :max="p.max" :step="p.step || (p.min != null && p.min < 0.01 ? 0.001 : 1)"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
              <input v-else type="text" :value="p.default"
                class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
              <p class="text-[10px] text-slate-400 mt-0.5">{{ p.desc }}</p>
            </div>
          </div>
          <div class="mt-4">
            <button class="px-5 py-2 bg-brand-600 text-white font-semibold text-sm rounded-lg hover:bg-brand-700 transition cursor-pointer shadow-sm">
              运行优化
            </button>
          </div>
        </div>
      </div>

      <!-- 方法对比 -->
      <div v-if="activeTab === 'compare'" class="space-y-4">
        <!-- 对比表格 -->
        <div class="bg-white rounded-xl border border-surface-3 overflow-hidden">
          <div class="px-5 py-3 border-b border-surface-3">
            <h3 class="text-sm font-semibold text-slate-600">优化方法对比</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="px-4 py-2.5">方法</th>
                  <th class="px-4 py-2.5 text-right">年化收益</th>
                  <th class="px-4 py-2.5 text-right">年化风险</th>
                  <th class="px-4 py-2.5 text-right">夏普比率</th>
                </tr>
              </thead>
              <tbody class="text-slate-700">
                <tr v-for="c in comparison" :key="c.method" class="border-b border-surface-2/60 hover:bg-brand-50/30 transition">
                  <td class="px-4 py-2.5 font-medium">{{ c.method }}</td>
                  <td class="px-4 py-2.5 text-right font-mono text-bull">+{{ c.return }}%</td>
                  <td class="px-4 py-2.5 text-right font-mono text-warn">{{ c.risk }}%</td>
                  <td class="px-4 py-2.5 text-right font-mono text-brand-600">{{ c.sharpe }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 图表 -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div class="bg-white rounded-xl border border-surface-3 p-4">
            <h3 class="text-sm font-semibold text-slate-600 mb-3">行业权重分布</h3>
            <div id="opt-weight-chart" class="w-full h-[280px]"></div>
          </div>
          <div class="bg-white rounded-xl border border-surface-3 p-4">
            <h3 class="text-sm font-semibold text-slate-600 mb-3">风险-收益散点</h3>
            <div id="opt-frontier-chart" class="w-full h-[280px]"></div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
