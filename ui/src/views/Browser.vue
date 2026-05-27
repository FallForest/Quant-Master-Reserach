<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { api, fmtNum } from '../utils/api'
import CandlestickChart from '../charts/CandlestickChart'

// ---- 状态 ----
const allStocks = ref([])
const searchQuery = ref('')
const page = ref(0)
const pageSize = 50
const selectedSymbol = ref(null)
const selectedName = ref('')
const showKline = ref(false)
const period = ref('D')
const loadingMin = ref(false)
const loadingList = ref(true)
const loadError = ref(false)
const lastUpdateDate = ref('--')
const sortField = ref('')
const sortAsc = ref(false)
const syncing = ref(false)
const syncError = ref('')

// K 线报价
const quote = ref({ close: 0, change: 0, changePct: '0' })

let chart = null
let rawDaily = []
let rawMin1 = []
let refreshTimer = null
let listRefreshTimer = null
let resizeObserver = null

const minutePeriods = ['1min']

// ---- 筛选 & 分页 ----
const filtered = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  let list = allStocks.value
  if (q) {
    list = list.filter(s =>
      s.symbol.toLowerCase().includes(q) || (s.name && s.name.toLowerCase().includes(q))
    )
  }
  if (sortField.value) {
    const f = sortField.value
    const dir = sortAsc.value ? 1 : -1
    list = [...list].sort((a, b) => {
      const va = a[f] ?? -Infinity
      const vb = b[f] ?? -Infinity
      return (va - vb) * dir
    })
  }
  return list
})

const paged = computed(() => {
  const start = page.value * pageSize
  return filtered.value.slice(start, start + pageSize)
})

const pageInfo = computed(() => {
  const start = page.value * pageSize + 1
  const end = Math.min((page.value + 1) * pageSize, filtered.value.length)
  return `${filtered.value.length ? start : 0}-${end} / ${filtered.value.length}`
})

function prevPage() { if (page.value > 0) page.value-- }
function nextPage() { if ((page.value + 1) * pageSize < filtered.value.length) page.value++ }

watch(searchQuery, () => { page.value = 0 })

// ---- 数据加载 ----
async function loadStocks() {
  loadingList.value = true
  loadError.value = false
  try {
    const [stockData, statusData] = await Promise.all([
      api('/api/browser/stocks'),
      api('/api/pipeline/status'),
    ])
    if (stockData?.stocks) {
      allStocks.value = stockData.stocks
    } else {
      loadError.value = true
    }
    if (statusData?.lastUpdate) {
      lastUpdateDate.value = statusData.lastUpdate
    }
    syncing.value = statusData?.syncing || false
    syncError.value = statusData?.syncError || ''
  } catch {
    loadError.value = true
  } finally {
    loadingList.value = false
  }
}

async function triggerSync() {
  syncing.value = true
  syncError.value = ''
  try {
    const resp = await api('/api/sync/trigger', { method: 'POST' })
    if (!resp?.ok) {
      syncError.value = resp?.error || '同步启动失败'
      syncing.value = false
    }
    // 轮询等待同步完成
    const poll = setInterval(async () => {
      const st = await api('/api/pipeline/status')
      if (!st?.syncing) {
        clearInterval(poll)
        syncing.value = false
        if (st?.syncError) syncError.value = st.syncError
        if (st?.lastUpdate) lastUpdateDate.value = st.lastUpdate
        loadStocks()
      }
    }, 3000)
  } catch {
    syncing.value = false
    syncError.value = '同步请求失败'
  }
}

onMounted(() => {
  loadStocks()
  // 列表行情每 3s 刷新
  listRefreshTimer = setInterval(async () => {
    const data = await api('/api/browser/quotes')
    if (!data?.quotes) return
    const q = data.quotes
    allStocks.value = allStocks.value.map(s => {
      const u = q[s.symbol]
      if (!u) return s
      return { ...s, close: u.close, change: u.change, changePct: u.changePct, volume: u.volume }
    })
  }, 3000)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  if (listRefreshTimer) clearInterval(listRefreshTimer)
  if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null }
  if (chart) { chart.destroy(); chart = null }
})

// ---- K 线查看 ----
let currentSymbol = ''

async function selectStock(symbol, name) {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null }
  selectedSymbol.value = symbol
  selectedName.value = name || ''
  showKline.value = true
  currentSymbol = symbol

  await nextTick()

  const dayData = await api(`/api/browser/kline/${symbol}?freq=1d`)
  if (currentSymbol !== symbol) return

  if (dayData?.kline?.length) {
    rawDaily = dayData.kline
    rawMin1 = []
    updateQuote(dayData.quote)
    if (minutePeriods.includes(period.value)) period.value = 'D'
    applyPeriod()
  } else {
    rawDaily = []
    rawMin1 = []
    quote.value = { close: 0, change: 0, changePct: '0' }
    applyPeriod()
  }

  // 启动实时刷新：3s 轮询行情 + 分时数据
  refreshTimer = setInterval(async () => {
    if (currentSymbol !== symbol) return
    const fresh = await api(`/api/realtime/kline/${symbol}`)
    if (currentSymbol !== symbol) return
    if (fresh?.quote) updateQuote(fresh.quote)
    if (minutePeriods.includes(period.value) && fresh?.kline?.length) {
      rawMin1 = fresh.kline
      applyPeriod()
    }
  }, 3000)
}

function updateQuote(q) {
  if (q) quote.value = q
}

function applyPeriod() {
  const p = period.value
  const isMinute = minutePeriods.includes(p)
  const source = isMinute ? rawMin1 : rawDaily

  if (source.length) {
    const last = source[source.length - 1]
    const prev = source.length > 1 ? source[source.length - 2] : null
    const change = prev ? +(last.close - prev.close).toFixed(2) : 0
    const changePct = prev && prev.close ? +(change / prev.close * 100).toFixed(2) : 0
    quote.value = { close: last.close, change, changePct: String(changePct) }
  }

  const data = (p === 'D' || p === '1min') ? source : aggregateKline(source, p)
  const container = document.getElementById('kline-chart')
  if (!container) return
  if (chart) chart.destroy()
  if (resizeObserver) resizeObserver.disconnect()
  chart = new CandlestickChart(container)
  chart.setPeriod(p)
  chart.setData(data)

  // 容器尺寸变化时自动 resize（处理 CSS 过渡动画）
  resizeObserver = new ResizeObserver(() => { if (chart) chart.resize() })
  resizeObserver.observe(container)
}

/** 日线 → 周/月聚合 */
function aggregateKline(data, mode) {
  if (!data.length) return []
  const groups = {}
  const getKey = mode === 'W'
    ? (d) => {
        const dt = new Date(d.date)
        const day = dt.getDay()
        const mon = new Date(dt)
        mon.setDate(dt.getDate() - ((day + 6) % 7))
        return mon.toISOString().slice(0, 10)
      }
    : (d) => d.date.slice(0, 7)

  data.forEach(d => {
    const key = getKey(d)
    if (!groups[key]) groups[key] = []
    groups[key].push(d)
  })

  return Object.keys(groups).sort().map(key => {
    const g = groups[key]
    return {
      date: g[g.length - 1].date,
      open: g[0].open,
      high: Math.max(...g.map(d => d.high)),
      low: Math.min(...g.map(d => d.low)),
      close: g[g.length - 1].close,
      volume: g.reduce((s, d) => s + d.volume, 0),
    }
  })
}

function closeKline() {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null }
  showKline.value = false
  selectedSymbol.value = null
  if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null }
  if (chart) { chart.destroy(); chart = null }
}

async function setPeriod(p) {
  period.value = p
  const isMinute = minutePeriods.includes(p)

  if (isMinute && selectedSymbol.value) {
    const sym = selectedSymbol.value
    loadingMin.value = true
    const minData = await api(`/api/realtime/kline/${sym}`)
    loadingMin.value = false
    if (currentSymbol !== sym) return
    rawMin1 = minData?.kline?.length ? minData.kline : []
    if (minData?.quote) updateQuote(minData.quote)
  }

  applyPeriod()
}

// 工具函数
function chgColor(val) { return val > 0 ? 'text-red-500' : val < 0 ? 'text-emerald-500' : 'text-slate-400' }

function toggleSort(field) {
  if (sortField.value === field) {
    sortAsc.value = !sortAsc.value
  } else {
    sortField.value = field
    sortAsc.value = false
  }
}

function sortIndicator(field) {
  if (sortField.value !== field) return ''
  return sortAsc.value ? ' ▲' : ' ▼'
}
</script>

<template>
  <div class="h-full flex flex-col animate-slide-in">

    <!-- 工具栏 -->
    <div class="flex-shrink-0 px-5 py-3 bg-white border-b border-slate-200/80 flex items-center gap-4">
      <div class="relative w-72">
        <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none"
             stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        <input v-model="searchQuery" type="text" placeholder="输入代码或名称搜索..."
          class="w-full pl-9 pr-4 py-2 text-sm rounded-xl border border-slate-200 bg-slate-50/50
                 focus:bg-white focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none transition-all duration-200">
      </div>
      <span class="inline-flex items-center gap-1.5 text-xs text-slate-500 bg-slate-100 px-2.5 py-1 rounded-full font-medium">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        {{ filtered.length }} 只股票
      </span>
      <span class="text-xs text-slate-400">数据截至 {{ lastUpdateDate }}</span>
      <button @click="triggerSync" :disabled="syncing"
        :class="['inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                 syncing ? 'bg-slate-100 text-slate-400 cursor-not-allowed' : 'bg-brand-50 text-brand-600 hover:bg-brand-100']">
        <svg v-if="syncing" class="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
        </svg>
        {{ syncing ? '同步中...' : '同步数据' }}
      </button>
      <span v-if="syncError" class="text-xs text-red-500">{{ syncError }}</span>
      <div class="flex-1"></div>
      <span v-if="showKline" class="text-xs text-slate-400 flex items-center gap-1.5">
        <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
        实时行情 · 3s 刷新
      </span>
    </div>

    <!-- 主体 -->
    <div class="flex-1 flex min-h-0 bg-slate-50/30">

      <!-- 股票列表 -->
      <div :class="['flex flex-col min-h-0 transition-all duration-300 bg-white',
                     showKline ? 'w-[420px] flex-shrink-0 border-r border-slate-200/80' : 'flex-1']">
        <div class="flex-1 overflow-auto">
          <table class="w-full text-sm">
            <thead class="sticky top-0 bg-white z-10">
              <tr class="text-left text-[11px] text-slate-400 uppercase tracking-wider border-b border-slate-100">
                <th class="px-4 py-2.5 w-[100px] font-semibold">代码</th>
                <th class="px-4 py-2.5 font-semibold">名称</th>
                <th @click="toggleSort('close')" class="px-4 py-2.5 text-right font-semibold cursor-pointer hover:text-slate-600 select-none">最新价{{ sortIndicator('close') }}</th>
                <th @click="toggleSort('changePct')" class="px-4 py-2.5 text-right w-[100px] font-semibold cursor-pointer hover:text-slate-600 select-none">涨跌幅{{ sortIndicator('changePct') }}</th>
                <th @click="toggleSort('volume')" class="px-4 py-2.5 text-right hidden xl:table-cell font-semibold cursor-pointer hover:text-slate-600 select-none">成交量{{ sortIndicator('volume') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="!paged.length">
                <td colspan="5" class="py-16 text-center">
                  <div v-if="loadingList" class="flex flex-col items-center gap-3">
                    <svg class="w-8 h-8 animate-spin text-brand-400" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                    </svg>
                    <span class="text-sm text-slate-400">加载中...</span>
                  </div>
                  <div v-else-if="loadError" class="flex flex-col items-center gap-3">
                    <svg class="w-10 h-10 text-slate-200" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                    <span class="text-sm text-slate-400">连接失败</span>
                    <button @click="loadStocks" class="px-4 py-1.5 text-xs font-semibold text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 cursor-pointer transition-colors">重试</button>
                  </div>
                  <div v-else class="flex flex-col items-center gap-3">
                    <svg class="w-10 h-10 text-slate-200" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    <span class="text-sm text-slate-400">{{ searchQuery ? '无匹配结果' : '暂无数据' }}</span>
                  </div>
                </td>
              </tr>
              <tr v-for="s in paged" :key="s.symbol"
                  @click="selectStock(s.symbol, s.name)"
                  :class="['cursor-pointer border-b border-slate-50 hover:bg-brand-50/50 transition-all duration-150 group',
                           selectedSymbol === s.symbol ? 'bg-brand-50/70' : '']">
                <td class="px-4 py-2.5 relative">
                  <div v-if="selectedSymbol === s.symbol" class="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-brand-500"></div>
                  <span class="font-mono text-brand-600 text-xs font-semibold tracking-wide">{{ s.symbol }}</span>
                </td>
                <td class="px-4 py-2.5 text-slate-700 font-medium truncate max-w-[120px]" :title="s.name">{{ s.name || '--' }}</td>
                <td :class="['px-4 py-2.5 text-right font-mono font-semibold text-sm', chgColor(s.change)]">
                  {{ s.close != null ? s.close.toFixed(2) : '--' }}
                </td>
                <td class="px-4 py-2.5 text-right">
                  <span v-if="s.changePct != null"
                    :class="['inline-flex items-center gap-0.5 px-2 py-0.5 rounded-md text-xs font-mono font-medium',
                             s.change > 0 ? 'bg-red-50 text-red-600' : s.change < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-50 text-slate-400']">
                    <svg v-if="s.change > 0" class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clip-rule="evenodd"/></svg>
                    <svg v-else-if="s.change < 0" class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                    {{ s.change > 0 ? '+' : '' }}{{ s.changePct }}%
                  </span>
                  <span v-else class="text-xs text-slate-400">--</span>
                </td>
                <td class="px-4 py-2.5 text-right font-mono text-xs text-slate-400 hidden xl:table-cell">
                  {{ fmtNum(s.volume) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- 分页 -->
        <div class="flex-shrink-0 px-4 py-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400 bg-white">
          <span>{{ pageInfo }}</span>
          <div class="flex gap-1.5">
            <button @click="prevPage" :disabled="page <= 0"
              class="px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 hover:border-slate-300 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium">上一页</button>
            <button @click="nextPage" :disabled="(page + 1) * pageSize >= filtered.length"
              class="px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 hover:border-slate-300 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium">下一页</button>
          </div>
        </div>
      </div>

      <!-- K 线查看器 -->
      <div v-show="showKline" class="flex-1 flex flex-col min-w-0 bg-white">
        <!-- 股票信息头 -->
        <div class="flex-shrink-0 px-5 py-3 border-b border-slate-200/80 flex items-center gap-5">
          <div class="flex items-center gap-2.5 min-w-0">
            <div class="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0">
              <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
              </svg>
            </div>
            <div class="min-w-0">
              <div class="flex items-baseline gap-2">
                <span class="text-sm font-bold text-slate-800">{{ selectedSymbol }}</span>
                <span class="text-xs text-slate-400 truncate">{{ selectedName }}</span>
              </div>
            </div>
          </div>
          <!-- 价格区 -->
          <div class="flex items-baseline gap-3">
            <span class="text-xl font-bold text-slate-900 font-mono tabular-nums">{{ quote.close ? quote.close.toFixed(2) : '--' }}</span>
            <div class="flex items-baseline gap-2">
              <span :class="['text-sm font-semibold font-mono tabular-nums', chgColor(quote.change)]">
                {{ quote.change > 0 ? '+' : '' }}{{ quote.change ? quote.change.toFixed(2) : '0.00' }}
              </span>
              <span :class="['text-xs font-semibold font-mono px-1.5 py-0.5 rounded',
                             quote.change > 0 ? 'bg-red-50 text-red-600' : quote.change < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-50 text-slate-400']">
                {{ quote.change > 0 ? '+' : '' }}{{ quote.changePct || '0' }}%
              </span>
            </div>
          </div>
          <div class="flex-1"></div>
          <!-- 周期按钮 -->
          <div class="flex items-center bg-slate-100 rounded-xl p-0.5 gap-0.5">
            <button @click="setPeriod('1min')"
              :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                       period === '1min' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
              1分
            </button>
            <span class="w-px h-4 bg-slate-200"></span>
            <button v-for="p in [{v:'D',l:'日K'},{v:'W',l:'周K'},{v:'M',l:'月K'}]" :key="p.v"
                    @click="setPeriod(p.v)"
              :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                       period === p.v ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
              {{ p.l }}
            </button>
          </div>
          <button @click="closeKline" class="p-1.5 rounded-lg hover:bg-slate-100 cursor-pointer transition-colors duration-150">
            <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
        <!-- 均线图例 -->
        <div class="flex-shrink-0 px-5 py-1.5 flex items-center gap-5 text-xs border-b border-slate-100">
          <span class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-[2px] rounded-full" style="background:#F59E0B"></span>
            <span class="text-slate-500 font-medium">MA5</span>
          </span>
          <span class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-[2px] rounded-full" style="background:#3B82F6"></span>
            <span class="text-slate-500 font-medium">MA10</span>
          </span>
          <span class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-[2px] rounded-full" style="background:#A855F7"></span>
            <span class="text-slate-500 font-medium">MA20</span>
          </span>
          <span class="flex-1"></span>
          <span v-if="loadingMin" class="text-brand-500 flex items-center gap-1.5">
            <svg class="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            加载中
          </span>
        </div>
        <!-- K 线图表 -->
        <div id="kline-chart" class="flex-1 min-h-0"></div>
      </div>
    </div>
  </div>
</template>
