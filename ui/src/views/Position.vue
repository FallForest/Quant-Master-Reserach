<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../utils/api'
import { fmtAmount, fmtPct, fmtPrice } from '../utils/format'
import { useKlineDetail } from '../composables/useKlineDetail'
import { useExecutionDraft } from '../composables/useExecutionDraft'

import PositionSummaryCards from '../components/position/PositionSummaryCards.vue'
import PositionHoldingsTable from '../components/position/PositionHoldingsTable.vue'
import PositionHoldingDetail from '../components/position/PositionHoldingDetail.vue'
import PositionAllocationCard from '../components/position/PositionAllocationCard.vue'
import PositionOrderHistoryCard from '../components/position/PositionOrderHistoryCard.vue'
import ExecutionWorkspace from '../components/position/ExecutionWorkspace.vue'
import TradeModal from '../components/position/TradeModal.vue'
import FeeSettingsModal from '../components/position/FeeSettingsModal.vue'

// ---- State ----
const loading = ref(true)
const positionData = ref(null)
const orders = ref([])
const ordersExpanded = ref(false)
const sortKey = ref('marketValue')
const sortDir = ref('desc')

// ---- K-line detail panel ----
const {
  selectedSymbol, selectedName, showKline, period, loadingMin, quote,
  marketOpen, checkMarketStatus,
  selectStock, setPeriod, closeDetail,
} = useKlineDetail({ chartId: 'position-kline-chart', mobileChartId: 'position-kline-chart-mobile' })

function normalizeSymbol(sym) {
  const s = (sym || '').trim().toUpperCase()
  if (/^(SH|SZ|BJ)/.test(s)) return s
  if (/^[689]/.test(s)) return 'SH' + s
  return 'SZ' + s
}

async function selectPosition(position) {
  const sym = normalizeSymbol(position?.instrument)
  if (!sym) return
  if (selectedSymbol.value === sym && showKline.value) {
    closeDetail()
    return
  }
  quote.value = {
    close: Number(position.currentPrice || 0),
    change: Number(position.pnl || 0),
    changePct: String(position.pnlPct || '0'),
  }
  await selectStock(sym, position.name || '')
}

// ---- Execution draft ----
const { draftMeta, draftTrades, importDraft, clearDraft } = useExecutionDraft()
const draftImported = ref(importDraft())

// ---- Trade modal ----
const tradeModalOpen = ref(false)
const tradePosition = ref(null)
const tradeModalRef = ref(null)

// ---- CRUD modal ----
const modalOpen = ref(false)
const modalMode = ref('add')
const modalForm = ref({ instrument: '', shares: '', price: '' })
const modalSaving = ref(false)
const modalError = ref('')
const deleteConfirm = ref('')

// ---- Fee settings ----
const feeSettingsOpen = ref(false)

// ---- Cash edit ----
const cashEditOpen = ref(false)
const cashForm = ref('')

// ---- Computed ----
const sortedPositions = computed(() => {
  if (!positionData.value?.positions) return []
  const arr = [...positionData.value.positions]
  arr.sort((a, b) => {
    const getVal = (x) => sortKey.value === 'buyFee' ? (x.buyFee?.total ?? 0) : (x[sortKey.value] ?? 0)
    const va = getVal(a)
    const vb = getVal(b)
    return sortDir.value === 'desc' ? vb - va : va - vb
  })
  return arr
})

// ---- Lifecycle ----
onMounted(async () => {
  loading.value = true
  await loadData()
  loading.value = false
})

// ---- Data loading ----
async function loadData() {
  const [posData, orderData] = await Promise.all([
    api('/api/positions'),
    api('/api/positions/history?limit=30'),
  ])
  if (posData && !posData.error) positionData.value = posData
  if (orderData?.orders) orders.value = orderData.orders
}

// ---- Sorting ----
function toggleSort(key) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'desc' ? 'asc' : 'desc'
  } else {
    sortKey.value = key
    sortDir.value = 'desc'
  }
}

// ---- Trade modal ----
function openTrade(position, side) {
  tradePosition.value = position
  tradeModalOpen.value = true
  nextTick(() => tradeModalRef.value?.open(side))
}

function closeTrade() {
  tradeModalOpen.value = false
  tradePosition.value = null
}

function onTradeSaved(data) {
  positionData.value = data
  loadData()
}

// ---- Position CRUD ----
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
  if (data?.error) { modalError.value = data.error }
  else if (data) { positionData.value = data; closeModal() }
}

async function deletePosition(instrument) {
  if (deleteConfirm.value !== instrument) {
    deleteConfirm.value = instrument
    setTimeout(() => { if (deleteConfirm.value === instrument) deleteConfirm.value = '' }, 3000)
    return
  }
  deleteConfirm.value = ''
  const data = await api(`/api/positions/${instrument}`, { method: 'DELETE' })
  if (data && !data.error) positionData.value = data
}

// ---- Cash edit ----
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
  if (data && !data.error) { positionData.value = data; cashEditOpen.value = false }
}

// ---- Execution complete ----
async function onExecutionComplete() {
  await Promise.all([loadData()])
}

// ---- Fee settings ----
function onFeeSaved(data) {
  positionData.value = data
  feeSettingsOpen.value = false
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">
    <!-- Header -->
    <div class="flex flex-wrap items-center gap-3">
      <h2 class="text-base font-semibold text-slate-700">交易执行</h2>
      <span v-if="positionData?.date" class="text-[11px] text-slate-400 font-mono">更新: {{ positionData.date }}</span>
      <div class="flex-1"></div>
      <button @click="openAddModal" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-cta text-white hover:bg-cta/90 active:bg-cta/80 transition cursor-pointer">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/></svg>
        添加持仓
      </button>
      <button @click="loadData" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800 transition cursor-pointer">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182"/></svg>
        刷新
      </button>
      <button @click="feeSettingsOpen = true" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium border-2 border-[#FF6600] text-[#FF6600] hover:bg-amber-50 active:bg-amber-100 transition cursor-pointer">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"/></svg>
        账户费率
      </button>
    </div>

    <!-- Loading -->
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

    <!-- Execution workspace -->
    <ExecutionWorkspace
      :draft-meta="draftMeta"
      :draft-orders="draftTrades"
      :initial-expanded="draftImported"
      @execution-complete="onExecutionComplete"
      @clear-draft="clearDraft"
    />

    <!-- Empty state -->
    <template v-if="!loading && (!positionData || positionData.positionCount === 0)">
      <div class="bg-white rounded-xl border border-surface-3 p-12 text-center">
        <svg class="w-16 h-16 mx-auto text-slate-300 mb-4" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3"/></svg>
        <h3 class="text-base font-semibold text-slate-600 mb-2">暂无持仓数据</h3>
        <p class="text-sm text-slate-400 mb-4">点击下方按钮手动添加持仓，或运行实时信号管道自动生成</p>
        <button @click="openAddModal" class="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium bg-cta text-white hover:bg-cta/90 active:bg-cta/80 transition cursor-pointer">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/></svg>
          添加持仓
        </button>
      </div>
    </template>

    <!-- Main content -->
    <template v-else-if="!loading">
      <PositionSummaryCards :position-data="positionData" @edit-cash="openCashEdit" />

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="lg:col-span-2 space-y-4">
          <PositionHoldingsTable
            :positions="sortedPositions"
            :sort-key="sortKey"
            :sort-dir="sortDir"
            :selected-symbol="selectedSymbol"
            :delete-confirm="deleteConfirm"
            @toggle-sort="toggleSort"
            @select-position="selectPosition"
            @position-keydown="handlePositionKeydown"
            @edit-position="openEditModal"
            @delete-position="deletePosition"
            @trade-position="p => openTrade(p, p.shares > 0 ? 'sell' : 'buy')"
          />
          <PositionOrderHistoryCard :orders="orders" :expanded="ordersExpanded" @toggle="ordersExpanded = !ordersExpanded" />
        </div>

        <div class="space-y-4">
          <PositionHoldingDetail
            v-show="showKline"
            class="hidden lg:block"
            :symbol="selectedSymbol"
            :name="selectedName"
            :quote="quote"
            :period="period"
            :loading-min="loadingMin"
            :market-open="marketOpen"
            chart-id="position-kline-chart"
            @close="closeDetail"
            @set-period="setPeriod"
          />
          <PositionAllocationCard :position-data="positionData" />
        </div>
      </div>
    </template>

    <!-- Mobile K-line overlay (not Teleported — in-DOM for reliable chart rendering) -->
    <div v-show="showKline" class="lg:hidden fixed inset-0 z-50 flex">
      <div class="absolute inset-0 bg-slate-950/45 backdrop-blur-sm" @click="closeDetail"></div>
      <div class="relative ml-auto h-full w-full max-w-xl bg-white shadow-2xl animate-slide-in flex min-h-0">
        <PositionHoldingDetail
          :symbol="selectedSymbol"
          :name="selectedName"
          :quote="quote"
          :period="period"
          :loading-min="loadingMin"
          :market-open="marketOpen"
          chart-id="position-kline-chart-mobile"
          @close="closeDetail"
          @set-period="setPeriod"
        />
      </div>
    </div>

    <!-- Add/Edit position modal -->
    <Teleport to="body">
      <div v-if="modalOpen" class="fixed inset-0 z-50 flex items-center justify-center">
        <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" @click="closeModal"></div>
        <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6 animate-slide-in">
          <button @click="closeModal" class="absolute top-4 right-4 p-1 rounded-lg hover:bg-surface-2 text-slate-400 hover:text-slate-600 transition cursor-pointer" aria-label="关闭">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
          </button>
          <h3 class="text-base font-semibold text-slate-800 mb-1">{{ modalMode === 'add' ? '添加持仓' : '编辑持仓' }}</h3>
          <p class="text-xs text-slate-400 mb-5">{{ modalMode === 'add' ? '输入股票代码、持仓数量和成本价格' : '修改 ' + modalForm.instrument + ' 的持仓信息' }}</p>
          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">股票代码</label>
              <input v-model="modalForm.instrument" :disabled="modalMode === 'edit'" type="text" placeholder="SH600011" :class="['w-full px-3 py-2.5 text-sm rounded-lg border bg-white outline-none transition font-mono', modalMode === 'edit' ? 'border-surface-3 text-slate-400 bg-surface-2/50 cursor-not-allowed' : 'border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500']" />
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
          <div v-if="modalError" class="mt-3 flex items-center gap-2 text-xs text-danger bg-danger/5 rounded-lg px-3 py-2">{{ modalError }}</div>
          <div class="flex items-center gap-3 mt-5">
            <button @click="closeModal" class="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium border border-surface-3 text-slate-600 hover:bg-surface-2 transition cursor-pointer">取消</button>
            <button @click="savePosition" :disabled="modalSaving" :class="['flex-1 flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer', modalSaving ? 'bg-surface-2 text-slate-400 cursor-not-allowed' : 'bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800']">
              {{ modalSaving ? '保存中...' : '确认' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Cash edit modal -->
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

    <!-- Trade modal -->
    <TradeModal
      v-if="tradeModalOpen"
      ref="tradeModalRef"
      :position="tradePosition"
      :fee-settings="positionData?.feeSettings"
      @close="closeTrade"
      @saved="onTradeSaved"
    />

    <!-- Fee settings modal -->
    <FeeSettingsModal
      v-if="feeSettingsOpen"
      :fee-settings="positionData?.feeSettings"
      :capital-amount="positionData?.capitalAmount"
      @close="feeSettingsOpen = false"
      @saved="onFeeSaved"
    />
  </div>
</template>
