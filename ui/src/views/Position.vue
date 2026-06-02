<script setup>
import { computed, ref, onMounted, onUnmounted, nextTick } from 'vue'
import { api } from '../utils/api'
import * as echarts from 'echarts'

// ---- 状态 ----
const loading = ref(true)
const positionData = ref(null)
const orders = ref([])
const ordersExpanded = ref(false)
const sortKey = ref('marketValue')
const sortDir = ref('desc')

// ---- 执行工作台 ----
const executionConfig = ref(null)
const executionPreview = ref(null)
const executionResults = ref(null)
const executionHistory = ref([])
const executionError = ref('')
const executionDraftMeta = ref(null)
const previewLoading = ref(false)
const submitting = ref(false)
const executionSubmitting = ref(false)
const executionForm = ref({
  brokerKind: 'paper',
  dryRun: true,
  maxOrderValue: '',
  maxPositionRatio: 1,
})

// 持仓编辑弹窗
const modalOpen = ref(false)
const modalMode = ref('add') // 'add' | 'edit'
const modalForm = ref({ instrument: '', shares: '', price: '' })
const modalSaving = ref(false)
const modalError = ref('')
const deleteConfirm = ref('') // 正在确认删除的 instrument

// 现金编辑
const cashEditOpen = ref(false)
const cashForm = ref('')

let pieChart = null

const previewOrders = computed(() => executionPreview.value?.orders || [])
const executionRuns = computed(() => executionHistory.value || [])
const executionExpanded = ref(false)

// ---- 数据加载 ----
onMounted(async () => {
  window.addEventListener('resize', handleResize)
  await loadPage()
})

onUnmounted(() => {
  pieChart?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  pieChart?.resize()
}

async function loadPage() {
  loading.value = true
  await Promise.all([loadData(), loadExecutionConfig(), loadExecutionHistory()])
  await importExecutionDraft()
  loading.value = false
}

async function loadData() {
  const [posData, orderData] = await Promise.all([
    api('/api/positions'),
    api('/api/positions/history?limit=30'),
  ])
  if (posData && !posData.error) {
    positionData.value = posData
    await nextTick()
    renderPieChart()
  }
  if (orderData?.orders) {
    orders.value = orderData.orders
  }
}

async function loadExecutionConfig() {
  const data = await api('/api/execution/config')
  if (!data || data.error) return
  executionConfig.value = data
  executionForm.value.brokerKind = data.defaultBrokerKind || 'paper'
  executionForm.value.dryRun = data.defaultDryRun !== false
  executionForm.value.maxPositionRatio = data.riskDefaults?.maxPositionRatio ?? 1
  executionForm.value.maxOrderValue = data.riskDefaults?.maxOrderValue ?? ''
}

async function loadExecutionHistory() {
  const data = await api('/api/execution/history?limit=30')
  if (data?.runs) executionHistory.value = data.runs
}

async function importExecutionDraft() {
  const raw = sessionStorage.getItem('executionDraft')
  if (!raw) return
  let draft
  try {
    draft = JSON.parse(raw)
  } catch {
    sessionStorage.removeItem('executionDraft')
    return
  }
  executionDraftMeta.value = {
    source: draft.source || 'buffered-rebalance',
    alias: draft.alias || '--',
    tradeDate: draft.tradeDate || '--',
    summary: draft.summary || null,
    config: draft.config || null,
  }
  await previewExecution(draft.trades || [])
}

async function previewExecution(trades) {
  previewLoading.value = true
  executionError.value = ''
  executionResults.value = null
  const payload = {
    trades,
    risk: buildRiskPayload(),
  }
  const data = await api('/api/execution/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!data || data.error) {
    executionError.value = data?.error || '生成执行预览失败'
    executionExpanded.value = true
    previewLoading.value = false
    return
  }
  executionPreview.value = data
  previewLoading.value = false
}

function buildRiskPayload() {
  return {
    maxOrderValue: executionForm.value.maxOrderValue === '' ? null : Number(executionForm.value.maxOrderValue),
    maxPositionRatio: Number(executionForm.value.maxPositionRatio || 1),
  }
}

async function submitExecution() {
  if (!previewOrders.value.length) {
    executionError.value = '没有可提交的执行订单'
    return
  }
  executionSubmitting.value = true
  executionError.value = ''
  const data = await api('/api/execution/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      brokerKind: executionForm.value.brokerKind,
      dryRun: executionForm.value.dryRun,
      risk: buildRiskPayload(),
      confirm: true,
      orders: previewOrders.value,
    }),
  })
  executionSubmitting.value = false
  if (!data || data.error) {
    executionError.value = data?.error || '提交执行失败'
    return
  }
  executionResults.value = data
  await Promise.all([loadExecutionHistory(), loadData()])
}

function clearExecutionDraft() {
  sessionStorage.removeItem('executionDraft')
  executionDraftMeta.value = null
  executionPreview.value = null
  executionResults.value = null
  executionError.value = ''
}

// ---- 排序 ----
function toggleSort(key) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'desc' ? 'asc' : 'desc'
  } else {
    sortKey.value = key
    sortDir.value = 'desc'
  }
}

function sortedPositions() {
  if (!positionData.value?.positions) return []
  const arr = [...positionData.value.positions]
  arr.sort((a, b) => {
    const va = a[sortKey.value] ?? 0
    const vb = b[sortKey.value] ?? 0
    return sortDir.value === 'desc' ? vb - va : va - vb
  })
  return arr
}

function sortIcon(key) {
  if (sortKey.value !== key) return ''
  return sortDir.value === 'desc' ? ' ↓' : ' ↑'
}

// ---- 图表 ----
function renderPieChart() {
  const el = document.getElementById('allocation-chart')
  if (!el || !positionData.value?.positions?.length) return
  pieChart?.dispose()
  pieChart = echarts.init(el)

  const positions = positionData.value.positions
  const cashWeight = positionData.value.totalAssets > 0
    ? (positionData.value.cash / positionData.value.totalAssets * 100)
    : 0

  const data = positions.map(p => ({
    name: p.name || p.instrument,
    value: p.marketValue,
  }))
  if (cashWeight > 0.5) {
    data.push({ name: '现金', value: positionData.value.cash })
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

// ---- 工具函数 ----
function fmtAmount(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  return Number(n).toLocaleString('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function fmtPct(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  const sign = n > 0 ? '+' : ''
  return sign + Number(n).toFixed(2) + '%'
}

function fmtPrice(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  return Number(n).toFixed(2)
}

function pnlClass(n) {
  if (n > 0) return 'text-bull'
  if (n < 0) return 'text-bear'
  return 'text-slate-500'
}

function orderSideLabel(side) {
  const map = { buy: '买入', sell: '卖出', close: '平仓' }
  return map[side] || side
}

function executionSideLabel(side) {
  return side === 'buy' ? '买入' : side === 'sell' ? '卖出' : side
}

function executionSideClass(side) {
  return side === 'buy'
    ? 'bg-bull/10 text-bull'
    : 'bg-bear/10 text-bear'
}

// ---- 持仓增删改 ----
function openAddModal() {
  modalMode.value = 'add'
  modalForm.value = { instrument: '', shares: '', price: '' }
  modalError.value = ''
  modalOpen.value = true
}

function openEditModal(p) {
  modalMode.value = 'edit'
  modalForm.value = { instrument: p.instrument, shares: String(p.shares), price: String(p.costPrice) }
  modalError.value = ''
  modalOpen.value = true
}

function closeModal() {
  modalOpen.value = false
  modalError.value = ''
}

async function savePosition() {
  const inst = modalForm.value.instrument.trim().toUpperCase()
  const shares = parseInt(modalForm.value.shares, 10)
  const price = parseFloat(modalForm.value.price)

  if (!inst) { modalError.value = '请输入股票代码'; return }
  if (!shares || shares <= 0) { modalError.value = '持仓数量必须大于 0'; return }
  if (!price || price <= 0) { modalError.value = '成本价格必须大于 0'; return }

  modalSaving.value = true
  modalError.value = ''

  let data
  if (modalMode.value === 'add') {
    data = await api('/api/positions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instrument: inst, shares, price }),
    })
  } else {
    data = await api(`/api/positions/${inst}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ shares, price }),
    })
  }

  modalSaving.value = false
  if (data?.error) {
    modalError.value = data.error
  } else if (data) {
    positionData.value = data
    await nextTick()
    renderPieChart()
    closeModal()
  }
}

async function deletePosition(instrument) {
  if (deleteConfirm.value !== instrument) {
    deleteConfirm.value = instrument
    setTimeout(() => { if (deleteConfirm.value === instrument) deleteConfirm.value = '' }, 3000)
    return
  }
  deleteConfirm.value = ''
  const data = await api(`/api/positions/${instrument}`, { method: 'DELETE' })
  if (data && !data.error) {
    positionData.value = data
    await nextTick()
    renderPieChart()
  }
}

// ---- 现金编辑 ----
function openCashEdit() {
  cashForm.value = String(Math.round(positionData.value?.cash || 0))
  cashEditOpen.value = true
}

async function saveCash() {
  const cash = parseFloat(cashForm.value)
  if (isNaN(cash) || cash < 0) return
  const data = await api('/api/positions/cash', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cash }),
  })
  if (data && !data.error) {
    positionData.value = data
    await nextTick()
    renderPieChart()
    cashEditOpen.value = false
  }
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">
    <div class="flex flex-wrap items-center gap-3">
      <h2 class="text-base font-semibold text-slate-700">交易执行</h2>
      <span v-if="positionData?.date" class="text-[11px] text-slate-400 font-mono">
        更新: {{ positionData.date }}
      </span>
      <div class="flex-1"></div>
      <button @click="openAddModal"
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
               bg-cta text-white hover:bg-cta/90 active:bg-cta/80 transition cursor-pointer">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
        </svg>
        添加持仓
      </button>
      <button @click="loadPage"
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
               bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800 transition cursor-pointer">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182"/>
        </svg>
        刷新
      </button>
    </div>

    <template v-if="loading">
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <div v-for="i in 5" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
          <div class="skeleton h-3 w-16 mb-2"></div>
          <div class="skeleton h-6 w-24 mb-1"></div>
          <div class="skeleton h-2.5 w-20"></div>
        </div>
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="lg:col-span-2 bg-white rounded-xl border border-surface-3 p-4">
          <div v-for="i in 5" :key="i" class="skeleton h-10 w-full rounded-lg mb-2"></div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <div class="skeleton w-full h-[280px] rounded-lg"></div>
        </div>
      </div>
    </template>

    <div class="bg-white rounded-xl border border-surface-3 p-4" data-testid="execution-workspace">
      <div class="flex flex-wrap items-center gap-3 justify-between">
        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <h3 class="text-sm font-semibold text-slate-700">执行工作台</h3>
            <span class="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] text-amber-700" data-testid="execution-safety-banner">
              {{ executionForm.brokerKind || 'paper' }} / {{ executionForm.dryRun ? 'dry-run' : 'live' }}
            </span>
            <span v-if="executionDraftMeta" class="rounded-full bg-brand-50 px-2 py-0.5 text-[11px] text-brand-700">
              已导入草案 · {{ previewOrders.length }} 笔
            </span>
            <span v-if="executionResults?.summary" class="rounded-full bg-success/10 px-2 py-0.5 text-[11px] text-success">
              accepted {{ executionResults.summary.accepted }}/{{ executionResults.summary.total }}
            </span>
          </div>
          <p class="mt-1 text-xs text-slate-500">
            持仓信息优先展示；需要下单时再展开执行面板。默认模拟提交，不会默认真实下单。
          </p>
        </div>
        <button
          class="inline-flex items-center gap-1.5 rounded-lg border border-surface-3 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-surface-2 transition cursor-pointer"
          @click="executionExpanded = !executionExpanded"
          data-testid="execution-toggle"
        >
          {{ executionExpanded ? '收起执行面板' : (executionDraftMeta ? '展开执行草案' : '展开执行面板') }}
          <svg :class="['w-4 h-4 text-slate-400 transition-transform', executionExpanded ? 'rotate-180' : '']" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5"/>
          </svg>
        </button>
      </div>

      <div v-if="executionDraftMeta && !executionExpanded" class="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-xs text-brand-700" data-testid="execution-draft-meta">
        <span class="font-medium">Buffered 草案</span>
        <span>来源：{{ executionDraftMeta.alias || '--' }}</span>
        <span>交易日：{{ executionDraftMeta.tradeDate || '--' }}</span>
        <span>预计买入：{{ fmtAmount(executionDraftMeta.summary?.estimatedBuyAmount) }}</span>
        <span>预计卖出：{{ fmtAmount(executionDraftMeta.summary?.estimatedSellAmount) }}</span>
      </div>

      <div v-if="executionError && !executionExpanded" class="mt-3 rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-sm text-danger" data-testid="execution-error">
        {{ executionError }}
      </div>

      <div v-if="executionExpanded" class="mt-4 space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          <div>
            <label class="block text-xs text-slate-500 mb-1.5">Broker mode</label>
            <select v-model="executionForm.brokerKind" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer" data-testid="execution-broker-kind">
              <option v-for="item in executionConfig?.supportedBrokers || ['paper']" :key="item" :value="item">{{ item }}</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1.5">dry-run</label>
            <label class="flex items-center gap-2 px-3 py-2 rounded-lg border border-surface-3 bg-white cursor-pointer min-h-[42px]">
              <input v-model="executionForm.dryRun" type="checkbox" class="accent-brand-600" data-testid="execution-dry-run" />
              <span class="text-sm text-slate-700">仅模拟提交</span>
            </label>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1.5">max order value</label>
            <input v-model="executionForm.maxOrderValue" type="number" min="0" step="1000" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" data-testid="execution-max-order-value" />
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1.5">max position ratio</label>
            <input v-model="executionForm.maxPositionRatio" type="number" min="0" max="1" step="0.01" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" data-testid="execution-max-position-ratio" />
          </div>
        </div>

        <div v-if="executionDraftMeta" class="rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-700" data-testid="execution-draft-meta">
          <div class="font-medium">已导入 Buffered 调仓草案</div>
          <div class="mt-1 text-xs">来源：{{ executionDraftMeta.alias || '--' }} · 交易日：{{ executionDraftMeta.tradeDate || '--' }}</div>
          <div class="mt-2 flex flex-wrap gap-2 text-xs text-brand-700">
            <span>预计买入：{{ fmtAmount(executionDraftMeta.summary?.estimatedBuyAmount) }}</span>
            <span>预计卖出：{{ fmtAmount(executionDraftMeta.summary?.estimatedSellAmount) }}</span>
            <span>预计费用：{{ fmtAmount(executionDraftMeta.summary?.estimatedFees) }}</span>
          </div>
        </div>

        <div v-if="executionError" class="rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-sm text-danger" data-testid="execution-error">
          {{ executionError }}
        </div>

        <div class="flex flex-wrap gap-2">
          <button
            class="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="previewLoading || !executionDraftMeta"
            @click="previewExecution(JSON.parse(sessionStorage.getItem('executionDraft') || '{}').trades || [])"
            data-testid="execution-preview-button"
          >
            {{ previewLoading ? '生成中...' : '重新生成执行预览' }}
          </button>
          <button
            class="px-4 py-2 rounded-lg bg-cta text-white text-sm font-medium hover:bg-cta/90 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="executionSubmitting || !previewOrders.length"
            @click="submitExecution"
            data-testid="execution-submit-button"
          >
            {{ executionSubmitting ? '提交中...' : '确认模拟提交' }}
          </button>
          <button
            class="px-4 py-2 rounded-lg border border-surface-3 text-sm font-medium text-slate-600 hover:bg-surface-2 transition cursor-pointer"
            @click="clearExecutionDraft"
          >
            清空草案
          </button>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-sm font-semibold text-slate-700">订单预览</h4>
              <div v-if="executionPreview?.summary" class="text-xs text-slate-500">
                valid {{ executionPreview.summary.validOrders }}/{{ executionPreview.summary.totalOrders }}
              </div>
            </div>
            <div v-if="!previewOrders.length" class="text-sm text-slate-400 py-8 text-center" data-testid="execution-preview-empty">暂无执行预览</div>
            <div v-else class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                    <th class="py-2 pr-3">代码</th>
                    <th class="py-2 pr-3">方向</th>
                    <th class="py-2 pr-3 text-right">价格</th>
                    <th class="py-2 pr-3 text-right">数量</th>
                    <th class="py-2 pr-3 text-right">金额</th>
                    <th class="py-2 pr-3">校验</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in previewOrders" :key="`${row.stockId}-${row.side}`" class="border-b border-surface-3/50 last:border-0">
                    <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.stockId }}</td>
                    <td class="py-2.5 pr-3">
                      <span :class="['inline-flex px-2 py-1 rounded-full text-[11px] font-medium', executionSideClass(row.side)]">{{ executionSideLabel(row.side) }}</span>
                    </td>
                    <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ fmtPrice(row.price) }}</td>
                    <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.amount.toLocaleString() }}</td>
                    <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ fmtAmount(row.orderValue) }}</td>
                    <td class="py-2.5 pr-3 text-xs">
                      <span v-if="row.valid" class="text-success">可提交</span>
                      <span v-else class="text-danger">{{ row.validationError }}</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-sm font-semibold text-slate-700">执行结果</h4>
              <div v-if="executionResults?.summary" class="text-xs text-slate-500">
                accepted {{ executionResults.summary.accepted }}/{{ executionResults.summary.total }}
              </div>
            </div>
            <div v-if="!executionResults?.results?.length" class="text-sm text-slate-400 py-8 text-center" data-testid="execution-results-empty">暂无执行结果</div>
            <div v-else class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                    <th class="py-2 pr-3">代码</th>
                    <th class="py-2 pr-3">方向</th>
                    <th class="py-2 pr-3">状态</th>
                    <th class="py-2 pr-3">post-check</th>
                    <th class="py-2 pr-3">备注</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in executionResults.results" :key="`${row.stockId}-${row.side}-${row.orderId || row.rejectionReason}`" class="border-b border-surface-3/50 last:border-0">
                    <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.stockId }}</td>
                    <td class="py-2.5 pr-3 text-xs text-slate-600">{{ executionSideLabel(row.side) }}</td>
                    <td class="py-2.5 pr-3 text-xs" :class="row.accepted ? 'text-success' : 'text-danger'">{{ row.accepted ? (row.status || 'accepted') : 'rejected' }}</td>
                    <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.postCheckStatus }}</td>
                    <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.rejectionReason || row.orderId || '--' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
          <h4 class="text-sm font-semibold text-slate-700 mb-3">执行历史</h4>
          <div v-if="!executionRuns.length" class="text-sm text-slate-400 py-6 text-center" data-testid="execution-history-empty">暂无执行历史</div>
          <div v-else class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">时间</th>
                  <th class="py-2 pr-3">broker</th>
                  <th class="py-2 pr-3">dry-run</th>
                  <th class="py-2 pr-3 text-right">accepted</th>
                  <th class="py-2 pr-3 text-right">rejected</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="run in executionRuns" :key="run.historyId" class="border-b border-surface-3/50 last:border-0">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ run.submittedAt }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ run.brokerKind }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ run.dryRun ? 'yes' : 'no' }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-success">{{ run.summary?.accepted ?? 0 }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-danger">{{ run.summary?.rejected ?? 0 }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <template v-if="!loading && (!positionData || positionData.positionCount === 0)">
      <div class="bg-white rounded-xl border border-surface-3 p-12 text-center">
        <svg class="w-16 h-16 mx-auto text-slate-300 mb-4" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3"/>
        </svg>
        <h3 class="text-base font-semibold text-slate-600 mb-2">暂无持仓数据</h3>
        <p class="text-sm text-slate-400 mb-4">
          点击下方按钮手动添加持仓，或运行实时信号管道自动生成
        </p>
        <button @click="openAddModal"
          class="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium
                 bg-cta text-white hover:bg-cta/90 active:bg-cta/80 transition cursor-pointer mb-4">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
          </svg>
          添加持仓
        </button>
        <div class="text-xs text-slate-400 bg-surface-2/50 rounded-lg p-3 max-w-md mx-auto text-left font-mono">
          { "cash": 1000000, "positions": { "SH600011": { "shares": 1000, "price": 8.5 } } }
        </div>
      </div>
    </template>

    <template v-else-if="!loading">
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">总资产</div>
          <div class="text-xl font-bold font-mono text-slate-800">{{ fmtAmount(positionData.totalAssets) }}</div>
          <div class="text-[10px] text-slate-400">持仓 + 现金</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">持仓市值</div>
          <div class="text-xl font-bold font-mono text-brand-600">{{ fmtAmount(positionData.totalMarketValue) }}</div>
          <div class="text-[10px] text-slate-400">{{ positionData.positionCount }} 只股票</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition cursor-pointer group"
             @click="openCashEdit">
          <div class="flex items-center gap-1">
            <div class="text-[11px] text-slate-500 mb-1">可用现金</div>
            <svg class="w-3 h-3 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"/>
            </svg>
          </div>
          <div class="text-xl font-bold font-mono text-slate-800">{{ fmtAmount(positionData.cash) }}</div>
          <div class="text-[10px] text-slate-400">点击编辑</div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">总盈亏</div>
          <div :class="['text-xl font-bold font-mono', pnlClass(positionData.totalPnl)]">
            {{ positionData.totalPnl > 0 ? '+' : '' }}{{ fmtAmount(positionData.totalPnl) }}
          </div>
          <div :class="['text-[10px]', pnlClass(positionData.totalPnlPct)]">
            {{ fmtPct(positionData.totalPnlPct) }}
          </div>
        </div>
        <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">仓位比例</div>
          <div class="text-xl font-bold font-mono text-slate-800">
            {{ positionData.totalAssets > 0 ? ((positionData.totalMarketValue / positionData.totalAssets) * 100).toFixed(1) : 0 }}%
          </div>
          <div class="text-[10px] text-slate-400">股票 / 总资产</div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="lg:col-span-2 bg-white rounded-xl border border-surface-3 p-4">
          <div class="flex items-center gap-2 mb-3">
            <h3 class="text-sm font-semibold text-slate-600">持仓明细</h3>
            <span class="text-[10px] text-slate-400">行内编辑 · 点击删除需二次确认</span>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-2 cursor-pointer select-none" @click="toggleSort('instrument')">代码{{ sortIcon('instrument') }}</th>
                  <th class="py-2 pr-2 cursor-pointer select-none" @click="toggleSort('name')">名称{{ sortIcon('name') }}</th>
                  <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="toggleSort('shares')">持仓{{ sortIcon('shares') }}</th>
                  <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="toggleSort('costPrice')">成本{{ sortIcon('costPrice') }}</th>
                  <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="toggleSort('currentPrice')">现价{{ sortIcon('currentPrice') }}</th>
                  <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="toggleSort('marketValue')">市值{{ sortIcon('marketValue') }}</th>
                  <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="toggleSort('pnlPct')">盈亏%{{ sortIcon('pnlPct') }}</th>
                  <th class="py-2 pl-2 w-20 cursor-pointer select-none" @click="toggleSort('weight')">占比{{ sortIcon('weight') }}</th>
                  <th class="py-2 pl-2 w-20 text-right">操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="p in sortedPositions()" :key="p.instrument" class="border-b border-surface-3/50 last:border-0 hover:bg-surface-2/30 transition">
                  <td class="py-2.5 pr-2 font-mono text-xs text-slate-700">{{ p.instrument }}</td>
                  <td class="py-2.5 pr-2 text-xs text-slate-600">{{ p.name }}</td>
                  <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ p.shares.toLocaleString() }}</td>
                  <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-500">{{ fmtPrice(p.costPrice) }}</td>
                  <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtPrice(p.currentPrice) }}</td>
                  <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtAmount(p.marketValue) }}</td>
                  <td :class="['py-2.5 pr-2 text-right font-mono text-xs', pnlClass(p.pnlPct)]">{{ fmtPct(p.pnlPct) }}</td>
                  <td class="py-2.5 pl-2">
                    <div class="flex items-center gap-1.5">
                      <div class="flex-1 bg-surface-2 rounded-full h-1.5">
                        <div class="h-1.5 rounded-full bg-brand-500/70 transition-all duration-300" :style="{ width: Math.min(p.weight, 100) + '%' }"></div>
                      </div>
                      <span class="text-[10px] font-mono text-slate-400 w-8 text-right">{{ p.weight }}%</span>
                    </div>
                  </td>
                  <td class="py-2.5 pl-2">
                    <div class="flex items-center justify-end gap-1">
                      <button @click.stop="openEditModal(p)" class="p-1 rounded hover:bg-brand-50 text-slate-400 hover:text-brand-600 transition cursor-pointer" aria-label="编辑">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"/>
                        </svg>
                      </button>
                      <button @click.stop="deletePosition(p.instrument)" :class="['p-1 rounded transition cursor-pointer', deleteConfirm === p.instrument ? 'bg-danger/10 text-danger' : 'hover:bg-danger/5 text-slate-400 hover:text-danger']" :aria-label="deleteConfirm === p.instrument ? '确认删除' : '删除'">
                        <svg v-if="deleteConfirm === p.instrument" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"/>
                        </svg>
                        <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/>
                        </svg>
                      </button>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">资产配置</h3>
          <div id="allocation-chart" class="w-full h-[280px]"></div>
        </div>
      </div>

      <div class="bg-white rounded-xl border border-surface-3 p-4">
        <button class="w-full flex items-center justify-between text-sm font-semibold text-slate-600 cursor-pointer" @click="ordersExpanded = !ordersExpanded">
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            历史委托
            <span class="text-[10px] font-normal text-slate-400">({{ orders.length }} 条)</span>
          </div>
          <svg :class="['w-4 h-4 text-slate-400 transition-transform', ordersExpanded ? 'rotate-180' : '']" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5"/>
          </svg>
        </button>
        <div v-if="ordersExpanded" class="mt-3 overflow-x-auto">
          <div v-if="!orders.length" class="text-center text-sm text-slate-400 py-6">暂无委托记录</div>
          <table v-else class="w-full text-sm">
            <thead>
              <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                <th class="py-2 pr-2">日期</th>
                <th class="py-2 pr-2">代码</th>
                <th class="py-2 pr-2">方向</th>
                <th class="py-2 pr-2 text-right">数量</th>
                <th class="py-2 pr-2 text-right">价格</th>
                <th class="py-2 pr-2 text-right">金额</th>
                <th class="py-2 pl-2">状态</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(o, i) in orders" :key="i" class="border-b border-surface-3/50 last:border-0">
                <td class="py-2 pr-2 font-mono text-xs text-slate-500">{{ o.date }}</td>
                <td class="py-2 pr-2 font-mono text-xs text-slate-700">{{ o.instrument || o.symbol }}</td>
                <td class="py-2 pr-2 text-xs">
                  <span :class="['px-1.5 py-0.5 rounded text-[10px] font-medium', (o.side === 'buy' || o.side === 'open') ? 'bg-bull/10 text-bull' : 'bg-bear/10 text-bear']">{{ orderSideLabel(o.side) }}</span>
                </td>
                <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ (o.shares || o.amount || 0).toLocaleString() }}</td>
                <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtPrice(o.price) }}</td>
                <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtAmount((o.shares || o.amount || 0) * (o.price || 0)) }}</td>
                <td class="py-2 pl-2 text-xs text-slate-500">{{ o.status || '已成交' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>

    <Teleport to="body">
      <div v-if="modalOpen" class="fixed inset-0 z-50 flex items-center justify-center">
        <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" @click="closeModal"></div>
        <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6 animate-slide-in">
          <button @click="closeModal" class="absolute top-4 right-4 p-1 rounded-lg hover:bg-surface-2 text-slate-400 hover:text-slate-600 transition cursor-pointer" aria-label="关闭">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>

          <h3 class="text-base font-semibold text-slate-800 mb-1">{{ modalMode === 'add' ? '添加持仓' : '编辑持仓' }}</h3>
          <p class="text-xs text-slate-400 mb-5">{{ modalMode === 'add' ? '输入股票代码、持仓数量和成本价格' : '修改 ' + modalForm.instrument + ' 的持仓信息' }}</p>

          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">股票代码</label>
              <input v-model="modalForm.instrument" :disabled="modalMode === 'edit'" type="text" placeholder="SH600011" :class="['w-full px-3 py-2.5 text-sm rounded-lg border bg-white outline-none transition font-mono', modalMode === 'edit' ? 'border-surface-3 text-slate-400 bg-surface-2/50 cursor-not-allowed' : 'border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500']" />
              <p class="text-[10px] text-slate-400 mt-1">格式：SH600000 或 SZ000001</p>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">持仓数量（股）</label>
              <input v-model="modalForm.shares" type="number" min="1" step="100" placeholder="1000" class="w-full px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">成本价格（元）</label>
              <input v-model="modalForm.price" type="number" min="0.01" step="0.01" placeholder="8.50" class="w-full px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" />
            </div>
          </div>

          <div v-if="modalForm.shares && modalForm.price" class="mt-4 flex items-center justify-between px-3 py-2.5 rounded-lg bg-surface-2/50 text-sm">
            <span class="text-slate-500">预估金额</span>
            <span class="font-mono font-semibold text-slate-700">{{ fmtAmount(parseInt(modalForm.shares || 0) * parseFloat(modalForm.price || 0)) }}</span>
          </div>

          <div v-if="modalError" class="mt-3 flex items-center gap-2 text-xs text-danger bg-danger/5 rounded-lg px-3 py-2">
            <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
            </svg>
            {{ modalError }}
          </div>

          <div class="flex items-center gap-3 mt-5">
            <button @click="closeModal" class="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium border border-surface-3 text-slate-600 hover:bg-surface-2 transition cursor-pointer">取消</button>
            <button @click="savePosition" :disabled="modalSaving" :class="['flex-1 flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer', modalSaving ? 'bg-surface-2 text-slate-400 cursor-not-allowed' : 'bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800']">
              <svg v-if="modalSaving" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              {{ modalSaving ? '保存中...' : '确认' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <Teleport to="body">
      <div v-if="cashEditOpen" class="fixed inset-0 z-50 flex items-center justify-center">
        <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" @click="cashEditOpen = false"></div>
        <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-sm mx-4 p-6 animate-slide-in">
          <h3 class="text-base font-semibold text-slate-800 mb-1">设置现金余额</h3>
          <p class="text-xs text-slate-400 mb-4">调整可用现金金额</p>
          <div class="flex items-center gap-2">
            <span class="text-sm text-slate-500">¥</span>
            <input v-model="cashForm" type="number" min="0" step="1000" class="flex-1 px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" />
          </div>
          <div class="flex items-center gap-3 mt-5">
            <button @click="cashEditOpen = false" class="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium border border-surface-3 text-slate-600 hover:bg-surface-2 transition cursor-pointer">取消</button>
            <button @click="saveCash" class="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800 transition cursor-pointer">确认</button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
