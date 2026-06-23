<script setup>
import { onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  positionData: {
    type: Object,
    default: null,
  },
})

let pieChart = null

function fmtAmount(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  return Number(n).toLocaleString('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function renderPieChart() {
  const el = document.getElementById('allocation-chart')
  if (!el || !props.positionData?.positions?.length) return
  pieChart?.dispose()
  pieChart = echarts.init(el)

  const positions = props.positionData.positions
  const cashWeight = props.positionData.totalAssets > 0
    ? (props.positionData.cash / props.positionData.totalAssets * 100)
    : 0

  const data = positions.map(p => ({
    name: p.name || p.instrument,
    value: p.marketValue,
  }))
  if (cashWeight > 0.5) {
    data.push({ name: '现金', value: props.positionData.cash })
  }

  pieChart.setOption({
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        return `<div style="font-weight:600">${params.name}</div>` +
               `<div>市值: <b>${fmtAmount(params.value)}</b></div>` +
               `<div>占比: ${params.percent.toFixed(1)}%</div>`
      },
    },
    legend: {
      orient: 'vertical',
      right: 10,
      top: 'center',
      textStyle: { color: '#64748B', fontSize: 11, fontFamily: 'Fira Sans' },
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 8,
    },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['35%', '50%'],
      avoidLabelOverlap: true,
      label: { show: false },
      emphasis: {
        label: { show: true, fontSize: 13, fontWeight: 'bold', color: '#0F172A' },
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' },
      },
      data,
      itemStyle: {
        borderRadius: 4,
        borderColor: '#fff',
        borderWidth: 2,
      },
    }],
    color: [
      '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
      '#EC4899', '#06B6D4', '#F97316', '#6366F1', '#14B8A6',
      '#84CC16', '#E11D48', '#A855F7', '#22D3EE', '#D946EF',
      '#64748B',
    ],
  })
}

onMounted(() => {
  renderPieChart()
  window.addEventListener('resize', resize)
})

watch(
  () => props.positionData,
  async () => {
    await nextTick()
    renderPieChart()
  },
  { deep: true },
)

onUnmounted(() => {
  pieChart?.dispose()
  window.removeEventListener('resize', resize)
})

function resize() {
  pieChart?.resize()
}
</script>

<template>
  <div class="bg-white rounded-xl border border-surface-3 p-4">
    <h3 class="text-sm font-semibold text-slate-600 mb-3">资产配置</h3>
    <div id="allocation-chart" class="w-full h-[280px]"></div>
  </div>
</template>
