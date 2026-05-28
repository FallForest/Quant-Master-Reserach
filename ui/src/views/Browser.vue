<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { api, fmtNum } from '../utils/api'
import CandlestickChart from '../charts/CandlestickChart'

// ---- 状态 ----
const allStocks = ref([])
const searchQuery = ref('')
const searchInput = ref(null)
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
const marketOpen = ref(false)

let chart = null
let rawDaily = []
let rawMin1 = []
let refreshTimer = null
let quoteRefreshTimer = null
let listRefreshTimer = null
let resizeObserver = null

const minutePeriods = ['1min']

function checkMarketStatus() {
  const now = new Date()
  const shanghaiMs = now.getTime() + 8 * 3600 * 1000
  const shanghai = new Date(shanghaiMs)
  const h = shanghai.getUTCHours()
  const m = shanghai.getUTCHours() * 60 + shanghai.getUTCMinutes()
  const dow = shanghai.getUTCDay()
  const weekday = dow >= 1 && dow <= 5
  const morning = m >= 570 && m < 690    // 9:30–11:30
  const afternoon = m >= 780 && m < 900  // 13:00–15:00
  marketOpen.value = weekday && (morning || afternoon)
}

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

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / pageSize)))

const paged = computed(() => {
  const start = page.value * pageSize
  return filtered.value.slice(start, start + pageSize)
})

const pageInfo = computed(() => {
  const start = page.value * pageSize + 1
  const end = Math.min((page.value + 1) * pageSize, filtered.value.length)
  return `${filtered.value.length ? start : 0}-${end} / ${filtered.value.length}`
})

// Numbered pagination: show up to 7 page buttons centered around current
const pageNumbers = computed(() => {
  const total = totalPages.value
  const cur = page.value
  if (total <= 7) return Array.from({ length: total }, (_, i) => i)

  const pages = [0]
  let start = Math.max(1, cur - 2)
  let end = Math.min(total - 2, cur + 2)
  if (cur < 3) end = Math.min(5, total - 2)
  if (cur > total - 4) start = Math.max(1, total - 6)

  if (start > 1) pages.push('...')
  for (let i = start; i <= end; i++) pages.push(i)
  if (end < total - 2) pages.push('...')
  pages.push(total - 1)
  return pages
})

const jumpPage = ref('')

function goToPage(p) {
  if (typeof p === 'number' && p >= 0 && p < totalPages.value) {
    page.value = p
  }
}

function handleJumpPage() {
  const p = parseInt(jumpPage.value, 10)
  if (!isNaN(p) && p >= 1 && p <= totalPages.value) {
    page.value = p - 1
  }
  jumpPage.value = ''
}

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

  // Keyboard shortcuts
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  if (quoteRefreshTimer) clearInterval(quoteRefreshTimer)
  if (listRefreshTimer) clearInterval(listRefreshTimer)
  if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null }
  if (chart) { chart.destroy(); chart = null }
  window.removeEventListener('keydown', handleKeydown)
})

function handleKeydown(e) {
  // Esc closes K-line panel
  if (e.key === 'Escape' && showKline.value) {
    closeKline()
    return
  }
  // / focuses search (when not in an input)
  if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
    const tag = e.target.tagName
    if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
      e.preventDefault()
      searchInput.value?.focus()
      return
    }
  }
  // Left/Right arrow for pagination (when not in search)
  if (document.activeElement !== searchInput.value) {
    if (e.key === 'ArrowLeft' && page.value > 0) {
      e.preventDefault()
      prevPage()
    } else if (e.key === 'ArrowRight' && (page.value + 1) * pageSize < filtered.value.length) {
      e.preventDefault()
      nextPage()
    }
  }
}

// ---- K 线查看 ----
let currentSymbol = ''

async function selectStock(symbol, name) {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null }
  if (quoteRefreshTimer) { clearInterval(quoteRefreshTimer); quoteRefreshTimer = null }
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

  // 启动实时刷新：3s 轮询分时数据
  refreshTimer = setInterval(async () => {
    if (currentSymbol !== symbol) return
    const fresh = await api(`/api/realtime/kline/${symbol}`)
    if (currentSymbol !== symbol) return
    if (minutePeriods.includes(period.value) && fresh?.kline?.length) {
      rawMin1 = fresh.kline
      applyPeriod()
    }
  }, 3000)

  // 启动实时报价轮询（TDX 实时价格，含 OHLCV）
  checkMarketStatus()
  quoteRefreshTimer = setInterval(async () => {
    if (currentSymbol !== symbol) return
    const q = await api(`/api/realtime/quote/${symbol}`)
    if (currentSymbol !== symbol) return
    if (q?.ok && q.quote) {
      const t = q.quote
      const lastClose = t.lastClose || 0
      const change = lastClose > 0 ? +(t.price - lastClose).toFixed(2) : 0
      const changePct = lastClose > 0 ? +(change / lastClose * 100).toFixed(2) : '0'
      quote.value = {
        close: t.price,
        change,
        changePct: String(changePct),
        open: t.open,
        high: t.high,
        low: t.low,
        volume: t.vol,
        amount: t.amount,
        lastClose,
      }
    }
    checkMarketStatus()
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
  if (quoteRefreshTimer) { clearInterval(quoteRefreshTimer); quoteRefreshTimer = null }
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
    <div class="flex-shrink-0 px-5 py-3 bg-white border-b border-surface-3 flex items-center gap-4">
      <div class="relative w-72">
        <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none"
             stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        <input ref="searchInput" v-model="searchQuery" type="text" placeholder="输入代码或名称搜索... 按 / 聚焦"
          aria-label="搜索股票"
          class="w-full pl-9 pr-4 py-2 text-sm rounded-xl border border-surface-3 bg-surface-2/30
                 focus:bg-white focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none transition-all duration-200">
      </div>
      <span class="inline-flex items-center gap-1.5 text-xs text-slate-500 bg-surface-2 px-2.5 py-1 rounded-full font-medium">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        {{ filtered.length }} 只股票
      </span>
      <span class="text-xs text-slate-400">数据截至 {{ lastUpdateDate }}</span>
      <button @click="triggerSync" :disabled="syncing"
        :aria-label="syncing ? '数据同步中' : '同步数据'"
        :class="['inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                 syncing ? 'bg-surface-2 text-slate-400 cursor-not-allowed' : 'bg-brand-50 text-brand-600 hover:bg-brand-100']">
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
    <div class="flex-1 flex min-h-0 bg-surface-0">

      <!-- 股票列表 -->
      <div :class="['flex flex-col min-h-0 transition-all duration-300 bg-white',
                     showKline ? 'w-[420px] flex-shrink-0 border-r border-surface-3 hidden lg:flex' : 'flex-1']">
        <div class="flex-1 overflow-auto">
          <table class="w-full text-sm">
            <thead class="sticky top-0 bg-white z-10">
              <tr class="text-left text-[11px] text-slate-500 uppercase tracking-wider border-b border-surface-3">
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
                  :class="['cursor-pointer border-b border-surface-2/60 hover:bg-brand-50/50 transition-all duration-150 group',
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
                             s.change > 0 ? 'bg-red-50 text-red-600' : s.change < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-surface-2 text-slate-400']">
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
        <!-- 分页 (numbered) -->
        <div class="flex-shrink-0 px-4 py-2.5 border-t border-surface-3 flex items-center justify-between text-xs text-slate-500 bg-white">
          <span>{{ pageInfo }}</span>
          <div class="flex items-center gap-1">
            <button @click="prevPage" :disabled="page <= 0" aria-label="上一页"
              class="px-2.5 py-1.5 rounded-lg border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/></svg>
            </button>
            <template v-for="p in pageNumbers" :key="p">
              <span v-if="p === '...'" class="px-1.5 text-slate-300">...</span>
              <button v-else @click="goToPage(p)"
                :class="['px-2.5 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all duration-150',
                         p === page ? 'bg-brand-500 text-white' : 'hover:bg-surface-2 text-slate-600 border border-surface-3']">
                {{ p + 1 }}
              </button>
            </template>
            <button @click="nextPage" :disabled="(page + 1) * pageSize >= filtered.length" aria-label="下一页"
              class="px-2.5 py-1.5 rounded-lg border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
            </button>
            <div class="flex items-center gap-1 ml-2 pl-2 border-l border-surface-3">
              <input v-model="jumpPage" @keydown.enter="handleJumpPage" type="number" min="1" :max="totalPages"
                placeholder="跳转" aria-label="跳转到页码"
                class="w-14 px-2 py-1 text-xs rounded-md border border-surface-3 focus:border-brand-400 outline-none text-center">
              <button @click="handleJumpPage" class="px-2 py-1 text-xs text-brand-600 hover:bg-brand-50 rounded-md cursor-pointer transition">GO</button>
            </div>
          </div>
        </div>
      </div>

      <!-- K 线查看器 (desktop: side panel, mobile/tablet: slide-over) -->
      <!-- Desktop side panel -->
      <div v-show="showKline" class="flex-1 flex-col min-w-0 bg-white hidden lg:flex">
        <KlineContent
          :symbol="selectedSymbol" :name="selectedName" :quote="quote" :period="period"
          :loadingMin="loadingMin" :marketOpen="marketOpen" @close="closeKline" @set-period="setPeriod"
        />
      </div>

      <!-- Mobile/Tablet slide-over -->
      <Teleport to="body">
        <div v-if="showKline" class="lg:hidden fixed inset-0 z-50 flex">
          <div class="absolute inset-0 bg-black/40" @click="closeKline"></div>
          <div class="relative ml-auto w-full max-w-[560px] bg-white shadow-2xl flex flex-col animate-slide-in-right">
            <KlineContent
              :symbol="selectedSymbol" :name="selectedName" :quote="quote" :period="period"
              :loadingMin="loadingMin" :marketOpen="marketOpen" @close="closeKline" @set-period="setPeriod"
            />
          </div>
        </div>
      </Teleport>
    </div>
  </div>
</template>

<!-- K-line content extracted to avoid duplication -->
<script>
import { defineComponent, h } from 'vue'

export const KlineContent = defineComponent({
  name: 'KlineContent',
  props: {
    symbol: String,
    name: String,
    quote: Object,
    period: String,
    loadingMin: Boolean,
    marketOpen: Boolean,
  },
  emits: ['close', 'set-period'],
  setup(props, { emit }) {
    function chgColor(val) { return val > 0 ? 'text-red-500' : val < 0 ? 'text-emerald-500' : 'text-slate-400' }
    function fmtVol(v) {
      if (!v) return '--'
      if (v >= 1e8) return (v / 1e8).toFixed(2) + '亿'
      if (v >= 1e4) return (v / 1e4).toFixed(1) + '万'
      return String(v)
    }
    function fmtAmt(v) {
      if (!v) return '--'
      if (v >= 1e8) return (v / 1e8).toFixed(2) + '亿'
      if (v >= 1e4) return (v / 1e4).toFixed(1) + '万'
      return String(Math.round(v))
    }

    return () => h('div', { class: 'flex-1 flex flex-col min-w-0 min-h-0' }, [
      // Header
      h('div', { class: 'flex-shrink-0 px-5 py-3 border-b border-surface-3 flex flex-wrap items-center gap-4' }, [
        h('div', { class: 'flex items-center gap-2.5 min-w-0' }, [
          h('div', { class: 'w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0' }, [
            h('svg', { class: 'w-4 h-4 text-brand-500', fill: 'none', stroke: 'currentColor', 'stroke-width': '2', viewBox: '0 0 24 24' }, [
              h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', d: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' }),
            ]),
          ]),
          h('div', { class: 'min-w-0' }, [
            h('div', { class: 'flex items-baseline gap-2' }, [
              h('span', { class: 'text-sm font-bold text-slate-800' }, props.symbol),
              h('span', { class: 'text-xs text-slate-400 truncate' }, props.name),
            ]),
          ]),
        ]),
        // Price
        h('div', { class: 'flex items-baseline gap-3' }, [
          h('span', { class: 'text-xl font-bold text-slate-900 font-mono tabular-nums' }, props.quote?.close?.toFixed(2) ?? '--'),
          h('div', { class: 'flex items-baseline gap-2' }, [
            h('span', { class: `text-sm font-semibold font-mono tabular-nums ${chgColor(props.quote?.change)}` },
              `${(props.quote?.change ?? 0) > 0 ? '+' : ''}${props.quote?.change?.toFixed(2) ?? '0.00'}`),
            h('span', {
              class: `text-xs font-semibold font-mono px-1.5 py-0.5 rounded ${
                (props.quote?.change ?? 0) > 0 ? 'bg-red-50 text-red-600' :
                (props.quote?.change ?? 0) < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-surface-2 text-slate-400'}`,
            }, `${(props.quote?.change ?? 0) > 0 ? '+' : ''}${props.quote?.changePct ?? '0'}%`),
          ]),
        ]),
        h('div', { class: 'flex-1' }),
        // Period buttons
        h('div', { class: 'flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5' }, [
          ...['1min', 'D', 'W', 'M'].map((p, i) => {
            const labels = { '1min': '1分', D: '日K', W: '周K', M: '月K' }
            return [
              i === 1 ? h('span', { class: 'w-px h-4 bg-surface-3' }) : null,
              h('button', {
                onClick: () => emit('set-period', p),
                'aria-label': `切换到${labels[p]}`,
                class: `px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200 ${
                  props.period === p ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`,
              }, labels[p]),
            ].filter(Boolean)
          }).flat(),
        ]),
        // Market status
        h('span', {
          class: `inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
            props.marketOpen ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`,
        }, [
          h('span', { class: `w-1.5 h-1.5 rounded-full ${props.marketOpen ? 'bg-emerald-400 animate-pulse' : 'bg-slate-300'}` }),
          props.marketOpen ? '交易中' : '闭市',
        ]),
        // Close
        h('button', { onClick: () => emit('close'), 'aria-label': '关闭K线', class: 'p-1.5 rounded-lg hover:bg-surface-2 cursor-pointer transition-colors duration-150' }, [
          h('svg', { class: 'w-4 h-4 text-slate-400', fill: 'none', stroke: 'currentColor', 'stroke-width': '2', viewBox: '0 0 24 24' }, [
            h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', d: 'M6 18L18 6M6 6l12 12' }),
          ]),
        ]),
      ]),
      // Real-time quote info bar (OHLCV)
      h('div', { class: 'flex-shrink-0 px-5 py-1.5 flex items-center gap-5 text-xs border-b border-surface-3 bg-surface-0/50' }, [
        h('span', { class: 'text-slate-400' }, '今开'),
        h('span', { class: 'font-mono text-slate-700 font-medium' }, props.quote?.open?.toFixed(2) ?? '--'),
        h('span', { class: 'text-slate-400' }, '最高'),
        h('span', { class: `font-mono font-medium ${chgColor((props.quote?.high ?? 0) - (props.quote?.lastClose ?? 0))}` }, props.quote?.high?.toFixed(2) ?? '--'),
        h('span', { class: 'text-slate-400' }, '最低'),
        h('span', { class: `font-mono font-medium ${chgColor((props.quote?.low ?? 0) - (props.quote?.lastClose ?? 0))}` }, props.quote?.low?.toFixed(2) ?? '--'),
        h('span', { class: 'text-slate-400' }, '昨收'),
        h('span', { class: 'font-mono text-slate-500' }, props.quote?.lastClose?.toFixed(2) ?? '--'),
        h('span', { class: 'text-slate-400' }, '成交量'),
        h('span', { class: 'font-mono text-slate-600' }, fmtVol(props.quote?.volume)),
        h('span', { class: 'text-slate-400' }, '成交额'),
        h('span', { class: 'font-mono text-slate-600' }, fmtAmt(props.quote?.amount)),
      ]),
      // MA legend
      h('div', { class: 'flex-shrink-0 px-5 py-1.5 flex items-center gap-5 text-xs border-b border-surface-3' }, [
        h('span', { class: 'flex items-center gap-1.5' }, [
          h('span', { class: 'inline-block w-3 h-[2px] rounded-full', style: 'background:#F59E0B' }),
          h('span', { class: 'text-slate-500 font-medium' }, 'MA5'),
        ]),
        h('span', { class: 'flex items-center gap-1.5' }, [
          h('span', { class: 'inline-block w-3 h-[2px] rounded-full', style: 'background:#3B82F6' }),
          h('span', { class: 'text-slate-500 font-medium' }, 'MA10'),
        ]),
        h('span', { class: 'flex items-center gap-1.5' }, [
          h('span', { class: 'inline-block w-3 h-[2px] rounded-full', style: 'background:#A855F7' }),
          h('span', { class: 'text-slate-500 font-medium' }, 'MA20'),
        ]),
        h('div', { class: 'flex-1' }),
        props.loadingMin ? h('span', { class: 'text-brand-500 flex items-center gap-1.5' }, [
          h('svg', { class: 'w-3.5 h-3.5 animate-spin', fill: 'none', viewBox: '0 0 24 24' }, [
            h('circle', { class: 'opacity-25', cx: '12', cy: '12', r: '10', stroke: 'currentColor', 'stroke-width': '4' }),
            h('path', { class: 'opacity-75', fill: 'currentColor', d: 'M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z' }),
          ]),
          '加载中',
        ]) : null,
      ]),
      // Chart area
      h('div', { id: 'kline-chart', class: 'flex-1 min-h-0' }),
    ])
  },
})
</script>

<style scoped>
@keyframes slide-in-right {
  from { transform: translateX(100%); }
  to { transform: translateX(0); }
}
.animate-slide-in-right { animation: slide-in-right 0.25s ease-out; }
</style>
