<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { api, fmtNum } from '../utils/api'
import * as echarts from 'echarts'

const loading = ref(true)
const portfolio = ref(null)
const sortField = ref('weight')
const sortAsc = ref(false)

let valueChart = null
let allocChart = null

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  loading.value = true
  const data = await api('/api/portfolio')
  portfolio.value = data
  loading.value = false
  await nextTick()
  renderCharts()
})

onUnmounted(() => {
  valueChart?.dispose()
  allocChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  valueChart?.resize()
  allocChart?.resize()
}

const sortedHoldings = () => {
  if (!portfolio.value?.holdings) return []
  const f = sortField.value
  const dir = sortAsc.value ? 1 : -1
  return [...portfolio.value.holdings].sort((a, b) => (a[f] - b[f]) * dir)
}

function toggleSort(field) {
  if (sortField.value === field) sortAsc.value = !sortAsc.value
  else { sortField.value = field; sortAsc.value = false }
}
function sortIndicator(field) {
  if (sortField.value !== field) return ''
  return sortAsc.value ? ' ▲' : ' ▼'
}

function renderCharts() {
  renderValueChart()
  renderAllocChart()
}

function renderValueChart() {
  const el = document.getElementById('portfolio-value-chart')
  if (!el || !portfolio.value?.timeline?.length) return
  valueChart?.dispose()
  valueChart = echarts.init(el)
  const tl = portfolio.value.timeline
  valueChart.setOption({
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const dt = params[0]?.axisValue || ''
        const val = params[0]?.value || 0
        return `<div style="font-weight:600;margin-bottom:4px">${dt}</div>净值: <b>¥${(val/10000).toFixed(2)}万</b>`
      },
    },
    grid: { left: 70, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'category', data: tl.map(d => d.date),
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: '组合净值 (万)',
      nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code', formatter: v => `${(v/10000).toFixed(0)}万` },
    },
    series: [{
      type: 'line', data: tl.map(d => d.value), smooth: true,
      lineStyle: { color: '#3B82F6', width: 2 }, itemStyle: { color: '#3B82F6' }, symbol: 'none',
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(59,130,246,0.2)' }, { offset: 1, color: 'rgba(59,130,246,0)' },
      ]) },
    }],
  })
}

function renderAllocChart() {
  const el = document.getElementById('portfolio-alloc-chart')
  if (!el || !portfolio.value?.allocation?.length) return
  allocChart?.dispose()
  allocChart = echarts.init(el)
  const colors = ['#3B82F6', '#EF4444', '#F59E0B', '#10B981', '#8B5CF6', '#EC4899', '#14B8A6', '#6366F1']
  allocChart.setOption({
    tooltip: {
      backgroundColor: 'rgba(15,23,42,0.92)', borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(p) { return `${p.name}<br/><b>${p.value}%</b>` },
    },
    series: [{
      type: 'pie', radius: ['40%', '70%'], center: ['50%', '50%'],
      label: { color: '#475569', fontSize: 11, formatter: '{b}\n{d}%' },
      labelLine: { lineStyle: { color: '#CBD5E1' } },
      data: portfolio.value.allocation.map((a, i) => ({
        name: a.sector, value: a.value,
        itemStyle: { color: colors[i % colors.length] },
      })),
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } },
    }],
  })
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 汇总卡片 -->
    <div v-if="loading" class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div v-for="i in 4" :key="i" class="bg-white rounded-xl border border-surface-3 p-4">
        <div class="skeleton h-3 w-16 mb-2"></div>
        <div class="skeleton h-7 w-24"></div>
      </div>
    </div>
    <div v-else-if="portfolio" class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition">
        <div class="text-xs text-slate-500 mb-1">组合总值</div>
        <div class="text-2xl font-bold font-mono text-slate-800">¥{{ (portfolio.summary.totalValue / 10000).toFixed(2) }}万</div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition">
        <div class="text-xs text-slate-500 mb-1">总盈亏</div>
        <div :class="['text-2xl font-bold font-mono', portfolio.summary.totalPnl >= 0 ? 'text-bull' : 'text-bear']">
          {{ portfolio.summary.totalPnl >= 0 ? '+' : '' }}¥{{ (portfolio.summary.totalPnl / 10000).toFixed(2) }}万
        </div>
        <div :class="['text-xs mt-0.5', portfolio.summary.totalPnlPct >= 0 ? 'text-bull' : 'text-bear']">
          {{ portfolio.summary.totalPnlPct >= 0 ? '+' : '' }}{{ portfolio.summary.totalPnlPct }}%
        </div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition">
        <div class="text-xs text-slate-500 mb-1">持仓数量</div>
        <div class="text-2xl font-bold font-mono text-brand-600">{{ portfolio.summary.stockCount }}只</div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4 hover:shadow-sm transition">
        <div class="text-xs text-slate-500 mb-1">行业分布</div>
        <div class="text-2xl font-bold font-mono text-brand-600">{{ portfolio.summary.sectorCount }}个</div>
      </div>
    </div>

    <!-- 净值曲线 + 行业配置 -->
    <div v-if="portfolio" class="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div class="lg:col-span-2 bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">组合净值走势</h3>
        <div id="portfolio-value-chart" class="w-full h-[280px]"></div>
      </div>
      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">行业配置</h3>
        <div id="portfolio-alloc-chart" class="w-full h-[280px]"></div>
      </div>
    </div>

    <!-- 持仓明细 -->
    <div v-if="portfolio" class="bg-white rounded-xl border border-surface-3 overflow-hidden">
      <div class="px-5 py-3 border-b border-surface-3">
        <h3 class="text-sm font-semibold text-slate-600">持仓明细</h3>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
              <th class="px-4 py-2.5">代码</th>
              <th class="px-4 py-2.5">名称</th>
              <th @click="toggleSort('weight')" class="px-4 py-2.5 text-right cursor-pointer hover:text-slate-600 select-none">权重{{ sortIndicator('weight') }}</th>
              <th @click="toggleSort('currentPrice')" class="px-4 py-2.5 text-right cursor-pointer hover:text-slate-600 select-none hidden sm:table-cell">现价{{ sortIndicator('currentPrice') }}</th>
              <th @click="toggleSort('costPrice')" class="px-4 py-2.5 text-right cursor-pointer hover:text-slate-600 select-none hidden md:table-cell">成本价{{ sortIndicator('costPrice') }}</th>
              <th @click="toggleSort('pnlPct')" class="px-4 py-2.5 text-right cursor-pointer hover:text-slate-600 select-none">盈亏{{ sortIndicator('pnlPct') }}</th>
            </tr>
          </thead>
          <tbody class="text-slate-700">
            <tr v-for="h in sortedHoldings()" :key="h.symbol" class="border-b border-surface-2/60 hover:bg-brand-50/30 transition">
              <td class="px-4 py-2.5 font-mono text-xs text-brand-600">{{ h.symbol }}</td>
              <td class="px-4 py-2.5">{{ h.name }}</td>
              <td class="px-4 py-2.5 text-right font-mono">{{ h.weight.toFixed(2) }}%</td>
              <td class="px-4 py-2.5 text-right font-mono hidden sm:table-cell">{{ h.currentPrice.toFixed(2) }}</td>
              <td class="px-4 py-2.5 text-right font-mono text-slate-400 hidden md:table-cell">{{ h.costPrice.toFixed(2) }}</td>
              <td :class="['px-4 py-2.5 text-right font-mono font-medium', h.pnlPct >= 0 ? 'text-bull' : 'text-bear']">
                {{ h.pnlPct >= 0 ? '+' : '' }}{{ h.pnlPct }}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
