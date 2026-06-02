<script setup>
import { ref, computed, onMounted, onUnmounted, onActivated, onDeactivated, nextTick, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, fmtNum } from '../utils/api'
import CandlestickChart from '../charts/CandlestickChart'
import KlineContent from '../components/KlineContent.vue'

const route = useRoute()
const router = useRouter()
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
const activeListMode = ref('all')
const watchlistSymbols = ref([])
const watchlistLoading = ref(false)
const watchlistError = ref('')
const watchlistPending = ref({})

const quote = ref({ close: 0, change: 0, changePct: '0' })
const marketOpen = ref(false)

const indices = ref([])

const watchlistSet = computed(() => new Set(watchlistSymbols.value))

const baseStocks = computed(() => {
  if (activeListMode.value === 'watchlist') {
    return allStocks.value.filter((stock) => watchlistSet.value.has(stock.symbol))
  }
  return allStocks.value
})

let chart = null
let rawDaily = []
let rawMin1 = []
let resizeObserver = null
let currentAbortController = null

let pagePollingStarted = false
let keydownBound = false

const pageTimers = new Set()
const detailTimers = new Set()
const minutePeriods = ['1min']
const realtimeDayRefreshMs = 5000

function isMinutePeriod(value) {
  return minutePeriods.includes(value)
}

function isDailyLikePeriod(value) {
  return ['D', 'W', 'M'].includes(value)
}

async function loadDailyKline(symbol, signal) {
  return api(`/api/browser/kline/${symbol}?freq=1d&includeRealtime=1`, { signal })
}

function setManagedInterval(timerSet, fn, ms) {
  const id = setInterval(fn, ms)
  timerSet.add(id)
  return id
}

function clearManagedTimers(timerSet) {
  for (const id of timerSet) {
    clearInterval(id)
  }
  timerSet.clear()
}

function checkMarketStatus() {
  const now = new Date()
  const shanghaiMs = now.getTime() + 8 * 3600 * 1000
  const shanghai = new Date(shanghaiMs)
  const minutes = shanghai.getUTCHours() * 60 + shanghai.getUTCMinutes()
  const dayOfWeek = shanghai.getUTCDay()
  const weekday = dayOfWeek >= 1 && dayOfWeek <= 5
  const morning = minutes >= 570 && minutes < 690
  const afternoon = minutes >= 780 && minutes < 900
  marketOpen.value = weekday && (morning || afternoon)
}

const filtered = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  let list = baseStocks.value
  if (q) {
    list = list.filter((stock) => (
      stock.symbol.toLowerCase().includes(q) ||
      (stock.name && stock.name.toLowerCase().includes(q))
    ))
  }
  if (sortField.value) {
    const field = sortField.value
    const direction = sortAsc.value ? 1 : -1
    list = [...list].sort((a, b) => {
      const va = a[field] ?? -Infinity
      const vb = b[field] ?? -Infinity
      return (va - vb) * direction
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

const pageNumbers = computed(() => {
  const total = totalPages.value
  const current = page.value
  if (total <= 7) return Array.from({ length: total }, (_, i) => i)

  const pages = [0]
  let start = Math.max(1, current - 2)
  let end = Math.min(total - 2, current + 2)
  if (current < 3) end = Math.min(5, total - 2)
  if (current > total - 4) start = Math.max(1, total - 6)

  if (start > 1) pages.push('...')
  for (let i = start; i <= end; i += 1) pages.push(i)
  if (end < total - 2) pages.push('...')
  pages.push(total - 1)
  return pages
})

const jumpPage = ref('')

function goToPage(target) {
  if (typeof target === 'number' && target >= 0 && target < totalPages.value) {
    page.value = target
  }
}

function handleJumpPage() {
  const target = parseInt(jumpPage.value, 10)
  if (!Number.isNaN(target) && target >= 1 && target <= totalPages.value) {
    page.value = target - 1
  }
  jumpPage.value = ''
}

function prevPage() {
  if (page.value > 0) page.value -= 1
}

function nextPage() {
  if ((page.value + 1) * pageSize < filtered.value.length) page.value += 1
}

watch(searchQuery, () => {
  page.value = 0
})

watch(activeListMode, () => {
  page.value = 0
})

watch(totalPages, (nextTotal) => {
  if (page.value >= nextTotal) {
    page.value = Math.max(0, nextTotal - 1)
  }
})

watch(() => route.query.symbol, async () => {
  await openStockFromRoute()
})

async function loadIndices() {
  const data = await api('/api/browser/indices')
  if (data?.indices) indices.value = data.indices
}

async function loadWatchlist() {
  watchlistLoading.value = true
  watchlistError.value = ''
  try {
    const data = await api('/api/watchlist')
    watchlistSymbols.value = Array.isArray(data?.symbols) ? data.symbols : []
  } catch {
    watchlistError.value = '自选列表加载失败'
  } finally {
    watchlistLoading.value = false
  }
}

function isWatchlisted(symbol) {
  return watchlistSet.value.has(symbol)
}

function setWatchlistPending(symbol, pending) {
  watchlistPending.value = {
    ...watchlistPending.value,
    [symbol]: pending,
  }
}

function resolveTargetStockSymbol() {
  return String(route.query.symbol || '').trim().toUpperCase()
}

async function openStockFromRoute() {
  const targetSymbol = resolveTargetStockSymbol()
  if (!targetSymbol || !allStocks.value.length) return

  const target = allStocks.value.find((stock) => stock.symbol === targetSymbol)
  if (!target) return

  searchQuery.value = target.symbol
  page.value = 0

  if (selectedSymbol.value !== target.symbol || !showKline.value) {
    await selectStock(target.symbol, target.name)
  }
}

async function navigateToStock(symbol) {
  if (!symbol) return
  const nextQuery = { ...route.query, symbol }
  await router.replace({ path: '/browser', query: nextQuery })
}

async function clearRouteSymbol() {
  if (!route.query.symbol) return
  const nextQuery = { ...route.query }
  delete nextQuery.symbol
  await router.replace({ path: '/browser', query: nextQuery })
}

async function toggleWatchlist(stock, event) {
  event?.stopPropagation?.()
  const symbol = stock?.symbol
  if (!symbol || watchlistPending.value[symbol]) return

  setWatchlistPending(symbol, true)
  watchlistError.value = ''
  try {
    const data = isWatchlisted(symbol)
      ? await api(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' })
      : await api('/api/watchlist', {
          method: 'POST',
          body: JSON.stringify({ symbol }),
        })
    watchlistSymbols.value = Array.isArray(data?.symbols) ? data.symbols : watchlistSymbols.value
  } catch {
    watchlistError.value = isWatchlisted(symbol) ? '移除自选失败' : '添加自选失败'
  } finally {
    setWatchlistPending(symbol, false)
  }
}

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
      await openStockFromRoute()
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
      return
    }

    const poll = setInterval(async () => {
      const status = await api('/api/pipeline/status')
      if (!status?.syncing) {
        clearInterval(poll)
        syncing.value = false
        if (status?.syncError) syncError.value = status.syncError
        if (status?.lastUpdate) lastUpdateDate.value = status.lastUpdate
        loadStocks()
      }
    }, 3000)
  } catch {
    syncing.value = false
    syncError.value = '同步请求失败'
  }
}

function bindKeydown() {
  if (keydownBound) return
  window.addEventListener('keydown', handleKeydown)
  keydownBound = true
}

function unbindKeydown() {
  if (!keydownBound) return
  window.removeEventListener('keydown', handleKeydown)
  keydownBound = false
}

function refreshPagedQuotes(quotes) {
  allStocks.value = allStocks.value.map((stock) => {
    const update = quotes[stock.symbol]
    if (!update) return stock
    return {
      ...stock,
      close: update.close,
      change: update.change,
      changePct: update.changePct,
      volume: update.volume,
    }
  })
}

function startPagePolling() {
  if (pagePollingStarted) return
  pagePollingStarted = true
  setManagedInterval(pageTimers, loadIndices, 1000)
  setManagedInterval(pageTimers, async () => {
    const symbols = paged.value.map((s) => s.symbol).join(',')
    if (!symbols) return
    const data = await api(`/api/browser/quotes?symbols=${symbols}`)
    if (!data?.quotes) return
    refreshPagedQuotes(data.quotes)
  }, 1000)
}

function stopPagePolling() {
  clearManagedTimers(pageTimers)
  pagePollingStarted = false
}

function stopDetailPolling() {
  clearManagedTimers(detailTimers)
}

onMounted(() => {
  loadStocks()
  loadWatchlist()
  loadIndices()
  startPagePolling()
  bindKeydown()
})

// keep-alive: 每次页面可见时启动轮询和键盘监听
onActivated(() => {
  startPagePolling()
  bindKeydown()
})

// keep-alive: 页面不可见时停止轮询，节省资源
onDeactivated(() => {
  stopPagePolling()
  stopDetailPolling()
  unbindKeydown()
})

// 组件真正销毁时的清理（keep-alive max 淘汰等场景）
onUnmounted(() => {
  stopPagePolling()
  stopDetailPolling()
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  if (chart) {
    chart.destroy()
    chart = null
  }
  unbindKeydown()
})

function handleKeydown(event) {
  if (event.key === 'Escape' && showKline.value) {
    closeKline()
    return
  }
  if (event.key === '/' && !event.ctrlKey && !event.metaKey) {
    const tag = event.target.tagName
    if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
      event.preventDefault()
      searchInput.value?.focus()
      return
    }
  }
  if (document.activeElement !== searchInput.value) {
    if (event.key === 'ArrowLeft' && page.value > 0) {
      event.preventDefault()
      prevPage()
    } else if (event.key === 'ArrowRight' && (page.value + 1) * pageSize < filtered.value.length) {
      event.preventDefault()
      nextPage()
    }
  }
}

async function selectStock(symbol, name) {
  await navigateToStock(symbol)

  // 取消上一次未完成的请求
  if (currentAbortController) {
    currentAbortController.abort()
  }
  currentAbortController = new AbortController()
  const signal = currentAbortController.signal

  stopDetailPolling()
  selectedSymbol.value = symbol
  selectedName.value = name || ''
  showKline.value = true

  await nextTick()

  const dayData = await loadDailyKline(symbol, signal)
  if (signal.aborted) return

  if (dayData?.kline?.length) {
    rawDaily = dayData.kline
    rawMin1 = []
    updateQuote(dayData.quote)
    if (isMinutePeriod(period.value)) period.value = 'D'
    applyPeriod()
  } else {
    rawDaily = []
    rawMin1 = []
    quote.value = { close: 0, change: 0, changePct: '0' }
    applyPeriod()
  }

  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedSymbol.value !== symbol || !isMinutePeriod(period.value)) return
    const fresh = await api(`/api/realtime/kline/${symbol}`, { signal })
    if (signal.aborted || selectedSymbol.value !== symbol) return
    if (fresh?.kline?.length) {
      rawMin1 = fresh.kline
      applyPeriod()
    }
  }, 1000)

  checkMarketStatus()
  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedSymbol.value !== symbol || !isDailyLikePeriod(period.value)) return
    const freshDaily = await loadDailyKline(symbol, signal)
    if (signal.aborted || selectedSymbol.value !== symbol) return
    if (freshDaily?.kline?.length) {
      rawDaily = freshDaily.kline
      if (freshDaily?.quote) updateQuote(freshDaily.quote)
      applyPeriod()
    }
  }, realtimeDayRefreshMs)

  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedSymbol.value !== symbol) return
    const realtime = await api(`/api/realtime/quote/${symbol}`, { signal })
    if (signal.aborted || selectedSymbol.value !== symbol) return
    if (realtime?.ok && realtime.quote) {
      const tick = realtime.quote
      const lastClose = tick.lastClose || 0
      const change = lastClose > 0 ? +(tick.price - lastClose).toFixed(2) : 0
      const changePct = lastClose > 0 ? +(change / lastClose * 100).toFixed(2) : '0'
      quote.value = {
        close: tick.price,
        change,
        changePct: String(changePct),
        open: tick.open,
        high: tick.high,
        low: tick.low,
        volume: tick.vol,
        amount: tick.amount,
        lastClose,
      }
    }
    checkMarketStatus()
  }, 1000)
}

function updateQuote(nextQuote) {
  if (nextQuote) quote.value = nextQuote
}

function applyPeriod() {
  const currentPeriod = period.value
  const isMinute = isMinutePeriod(currentPeriod)
  const source = isMinute ? rawMin1 : rawDaily

  if (source.length) {
    const last = source[source.length - 1]
    const prev = source.length > 1 ? source[source.length - 2] : null
    const change = prev ? +(last.close - prev.close).toFixed(2) : 0
    const changePct = prev && prev.close ? +(change / prev.close * 100).toFixed(2) : 0
    quote.value = { close: last.close, change, changePct: String(changePct) }
  }

  const data = (currentPeriod === 'D' || currentPeriod === '1min')
    ? source
    : aggregateKline(source, currentPeriod)
  const container = document.getElementById('kline-chart')
  if (!container) return
  if (chart) chart.destroy()
  if (resizeObserver) resizeObserver.disconnect()
  chart = new CandlestickChart(container)
  chart.setPeriod(currentPeriod)
  chart.setData(data)

  resizeObserver = new ResizeObserver(() => {
    if (chart) chart.resize()
  })
  resizeObserver.observe(container)
}

function aggregateKline(data, mode) {
  if (!data.length) return []
  const groups = {}
  const getKey = mode === 'W'
    ? (item) => {
        const date = new Date(item.date)
        const day = date.getDay()
        const monday = new Date(date)
        monday.setDate(date.getDate() - ((day + 6) % 7))
        return monday.toISOString().slice(0, 10)
      }
    : (item) => item.date.slice(0, 7)

  data.forEach((item) => {
    const key = getKey(item)
    if (!groups[key]) groups[key] = []
    groups[key].push(item)
  })

  return Object.keys(groups).sort().map((key) => {
    const group = groups[key]
    return {
      date: group[group.length - 1].date,
      open: group[0].open,
      high: Math.max(...group.map((item) => item.high)),
      low: Math.min(...group.map((item) => item.low)),
      close: group[group.length - 1].close,
      volume: group.reduce((sum, item) => sum + item.volume, 0),
    }
  })
}

async function closeKline() {
  if (currentAbortController) {
    currentAbortController.abort()
    currentAbortController = null
  }
  stopDetailPolling()
  showKline.value = false
  selectedSymbol.value = null
  selectedName.value = ''
  await clearRouteSymbol()
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  if (chart) {
    chart.destroy()
    chart = null
  }
}

async function setPeriod(nextPeriod) {
  period.value = nextPeriod
  const isMinute = isMinutePeriod(nextPeriod)

  if (isMinute && selectedSymbol.value) {
    const symbol = selectedSymbol.value
    loadingMin.value = true
    const minuteData = await api(`/api/realtime/kline/${symbol}`)
    loadingMin.value = false
    if (selectedSymbol.value !== symbol) return
    rawMin1 = minuteData?.kline?.length ? minuteData.kline : []
    if (minuteData?.quote) updateQuote(minuteData.quote)
  }

  applyPeriod()
}

function chgColor(value) {
  return value > 0 ? 'text-red-500' : value < 0 ? 'text-emerald-500' : 'text-slate-400'
}

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
    <div class="flex-shrink-0 px-5 py-3 bg-white border-b border-surface-3 flex items-center gap-4">
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
        <button
          :class="[
            'px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-100',
            activeListMode === 'all' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700',
          ]"
          type="button"
          @click="activeListMode = 'all'"
        >
          全部股票
        </button>
        <button
          :class="[
            'px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-100',
            activeListMode === 'watchlist' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700',
          ]"
          type="button"
          @click="activeListMode = 'watchlist'"
        >
          自选股票
        </button>
      </div>
      <span class="inline-flex items-center gap-1.5 text-xs text-slate-500 bg-amber-50 px-2.5 py-1 rounded-full font-medium border border-amber-100">
        <svg class="w-3.5 h-3.5 text-amber-400" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.176 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81H7.03a1 1 0 00.951-.69l1.07-3.292z"/>
        </svg>
        {{ watchlistSymbols.length }} 自选
      </span>
      <div class="relative w-72">
        <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        <input
          ref="searchInput"
          v-model="searchQuery"
          aria-label="搜索股票"
          class="w-full pl-9 pr-4 py-2 text-sm rounded-xl border border-surface-3 bg-surface-2/30 focus:bg-white focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none transition-all duration-200"
          placeholder="输入代码或名称搜索，按 / 聚焦"
          type="text"
        >
      </div>
      <span class="inline-flex items-center gap-1.5 text-xs text-slate-500 bg-surface-2 px-2.5 py-1 rounded-full font-medium">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        {{ filtered.length }} 只股票
      </span>
      <span class="text-xs text-slate-400">数据截至 {{ lastUpdateDate }}</span>
      <button
        :aria-label="syncing ? '同步中' : '同步数据'"
        :class="[
          'inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
          syncing ? 'bg-surface-2 text-slate-400 cursor-not-allowed' : 'bg-brand-50 text-brand-600 hover:bg-brand-100',
        ]"
        :disabled="syncing"
        @click="triggerSync"
      >
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
      <span v-else-if="watchlistError" class="text-xs text-red-500">{{ watchlistError }}</span>
      <span v-else-if="watchlistLoading" class="text-xs text-slate-400">自选加载中...</span>
      <div class="flex-1"></div>
      <span v-if="showKline" class="text-xs text-slate-400 flex items-center gap-1.5">
        <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
        实时行情 · 1s 刷新
      </span>
    </div>

    <!-- 大盘指数看板 -->
    <div v-if="indices.length" class="flex-shrink-0 bg-surface-0 border-b border-surface-3">
      <div class="px-5 py-3 flex items-stretch gap-3 overflow-x-auto scrollbar-none">
        <div
          v-for="idx in indices"
          :key="idx.symbol"
          :class="[
            'flex-shrink-0 flex flex-col justify-between min-w-[140px] px-4 py-2.5 rounded-xl border transition-all duration-200 cursor-pointer',
            idx.change > 0
              ? 'bg-red-50/60 border-red-100 hover:border-red-200 hover:shadow-sm'
              : idx.change < 0
                ? 'bg-emerald-50/60 border-emerald-100 hover:border-emerald-200 hover:shadow-sm'
                : 'bg-white border-surface-3 hover:border-slate-300 hover:shadow-sm',
          ]"
        >
          <span class="text-[11px] text-slate-500 font-medium truncate">{{ idx.name }}</span>
          <span :class="['text-base font-bold font-mono tabular-nums leading-tight', chgColor(idx.change)]">
            {{ idx.price ? idx.price.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--' }}
          </span>
          <div class="flex items-center gap-1.5 mt-1">
            <span :class="['text-xs font-medium font-mono', chgColor(idx.change)]">
              {{ idx.change > 0 ? '+' : '' }}{{ idx.change != null ? idx.change.toFixed(2) : '--' }}
            </span>
            <span class="text-slate-300">|</span>
            <span
              :class="[
                'text-xs font-semibold font-mono px-1.5 py-0.5 rounded-md',
                idx.change > 0 ? 'bg-red-100 text-red-600' : idx.change < 0 ? 'bg-emerald-100 text-emerald-600' : 'bg-slate-100 text-slate-400',
              ]"
            >
              {{ idx.change > 0 ? '+' : '' }}{{ idx.changePct }}%
            </span>
          </div>
        </div>
      </div>
    </div>

    <div class="flex-1 flex min-h-0 bg-surface-0">
      <div :class="['flex flex-col min-h-0 transition-all duration-300 bg-white', showKline ? 'w-[420px] flex-shrink-0 border-r border-surface-3 hidden lg:flex' : 'flex-1']">
        <div class="flex-1 overflow-auto">
          <table class="w-full text-sm">
            <thead class="sticky top-0 bg-white z-10">
              <tr class="text-left text-[11px] text-slate-500 uppercase tracking-wider border-b border-surface-3">
                <th class="px-3 py-2.5 w-[52px] font-semibold text-center">自选</th>
                <th class="px-4 py-2.5 w-[100px] font-semibold">代码</th>
                <th class="px-4 py-2.5 font-semibold">名称</th>
                <th class="px-4 py-2.5 text-right font-semibold cursor-pointer hover:text-slate-600 select-none" @click="toggleSort('close')">最新价{{ sortIndicator('close') }}</th>
                <th class="px-4 py-2.5 text-right w-[100px] font-semibold cursor-pointer hover:text-slate-600 select-none" @click="toggleSort('changePct')">涨跌幅{{ sortIndicator('changePct') }}</th>
                <th class="px-4 py-2.5 text-right hidden xl:table-cell font-semibold cursor-pointer hover:text-slate-600 select-none" @click="toggleSort('volume')">成交量{{ sortIndicator('volume') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="!paged.length">
                <td colspan="6" class="py-16 text-center">
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
                    <button class="px-4 py-1.5 text-xs font-semibold text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 cursor-pointer transition-colors" @click="loadStocks">重试</button>
                  </div>
                  <div v-else class="flex flex-col items-center gap-3">
                    <svg class="w-10 h-10 text-slate-200" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    <span class="text-sm text-slate-400">{{ searchQuery ? '无匹配结果' : activeListMode === 'watchlist' ? '暂无自选股票，可在全部股票中点击星标添加' : '暂无数据' }}</span>
                  </div>
                </td>
              </tr>
              <tr
                v-for="stock in paged"
                :key="stock.symbol"
                :class="['cursor-pointer border-b border-surface-2/60 hover:bg-brand-50/50 transition-all duration-150 group', selectedSymbol === stock.symbol ? 'bg-brand-50/70' : '']"
                @click="selectStock(stock.symbol, stock.name)"
              >
                <td class="px-3 py-2.5 text-center">
                  <button
                    :aria-label="isWatchlisted(stock.symbol) ? '从自选移除' : '添加到自选'"
                    :class="[
                      'inline-flex items-center justify-center w-8 h-8 rounded-lg border transition-all duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-100',
                      isWatchlisted(stock.symbol)
                        ? 'border-amber-200 bg-amber-50 text-amber-500 hover:bg-amber-100'
                        : 'border-surface-3 text-slate-300 hover:text-amber-400 hover:border-amber-200 hover:bg-amber-50/60',
                      watchlistPending[stock.symbol] ? 'opacity-60 cursor-wait' : '',
                    ]"
                    :disabled="watchlistPending[stock.symbol]"
                    type="button"
                    @click="toggleWatchlist(stock, $event)"
                  >
                    <svg class="w-4 h-4" :fill="isWatchlisted(stock.symbol) ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321 1.01l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.386a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.56a.562.562 0 01-.84-.61l1.285-5.386a.563.563 0 00-.182-.557L3.04 10.407a.563.563 0 01.321-1.01l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/>
                    </svg>
                  </button>
                </td>
                <td class="px-4 py-2.5 relative">
                  <div v-if="selectedSymbol === stock.symbol" class="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-brand-500"></div>
                  <span class="font-mono text-brand-600 text-xs font-semibold tracking-wide">{{ stock.symbol }}</span>
                </td>
                <td class="px-4 py-2.5 text-slate-700 font-medium truncate max-w-[120px]" :title="stock.name">{{ stock.name || '--' }}</td>
                <td :class="['px-4 py-2.5 text-right font-mono font-semibold text-sm', chgColor(stock.change)]">
                  {{ stock.close != null ? stock.close.toFixed(2) : '--' }}
                </td>
                <td class="px-4 py-2.5 text-right">
                  <span
                    v-if="stock.changePct != null"
                    :class="[
                      'inline-flex items-center gap-0.5 px-2 py-0.5 rounded-md text-xs font-mono font-medium',
                      stock.change > 0 ? 'bg-red-50 text-red-600' : stock.change < 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-surface-2 text-slate-400',
                    ]"
                  >
                    <svg v-if="stock.change > 0" class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clip-rule="evenodd"/></svg>
                    <svg v-else-if="stock.change < 0" class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                    {{ stock.change > 0 ? '+' : '' }}{{ stock.changePct }}%
                  </span>
                  <span v-else class="text-xs text-slate-400">--</span>
                </td>
                <td class="px-4 py-2.5 text-right font-mono text-xs text-slate-400 hidden xl:table-cell">
                  {{ fmtNum(stock.volume) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="flex-shrink-0 px-4 py-2.5 border-t border-surface-3 flex items-center justify-between text-xs text-slate-500 bg-white">
          <span>{{ pageInfo }}</span>
          <div class="flex items-center gap-1">
            <button
              aria-label="上一页"
              class="px-2.5 py-1.5 rounded-lg border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium"
              :disabled="page <= 0"
              @click="prevPage"
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/></svg>
            </button>
            <template v-for="pageNo in pageNumbers" :key="pageNo">
              <span v-if="pageNo === '...'" class="px-1.5 text-slate-300">...</span>
              <button
                v-else
                :class="[
                  'px-2.5 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all duration-150',
                  pageNo === page ? 'bg-brand-500 text-white' : 'hover:bg-surface-2 text-slate-600 border border-surface-3',
                ]"
                @click="goToPage(pageNo)"
              >
                {{ pageNo + 1 }}
              </button>
            </template>
            <button
              aria-label="下一页"
              class="px-2.5 py-1.5 rounded-lg border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 text-slate-600 font-medium"
              :disabled="(page + 1) * pageSize >= filtered.length"
              @click="nextPage"
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
            </button>
            <div class="flex items-center gap-1 ml-2 pl-2 border-l border-surface-3">
              <input
                v-model="jumpPage"
                aria-label="跳转到页码"
                class="w-14 px-2 py-1 text-xs rounded-md border border-surface-3 focus:border-brand-400 outline-none text-center"
                :max="totalPages"
                min="1"
                placeholder="跳转"
                type="number"
                @keydown.enter="handleJumpPage"
              >
              <button class="px-2 py-1 text-xs text-brand-600 hover:bg-brand-50 rounded-md cursor-pointer transition" @click="handleJumpPage">GO</button>
            </div>
          </div>
        </div>
      </div>

      <div v-show="showKline" class="flex-1 flex-col min-w-0 bg-white hidden lg:flex">
        <KlineContent
          :symbol="selectedSymbol"
          :name="selectedName"
          :quote="quote"
          :period="period"
          :loadingMin="loadingMin"
          :marketOpen="marketOpen"
          @close="closeKline"
          @set-period="setPeriod"
        />
      </div>

      <Teleport to="body">
        <div v-if="showKline" class="lg:hidden fixed inset-0 z-50 flex">
          <div class="absolute inset-0 bg-black/40" @click="closeKline"></div>
          <div class="relative ml-auto w-full max-w-[560px] bg-white shadow-2xl flex flex-col animate-slide-in-right">
            <KlineContent
              :symbol="selectedSymbol"
              :name="selectedName"
              :quote="quote"
              :period="period"
              :loadingMin="loadingMin"
              :marketOpen="marketOpen"
              @close="closeKline"
              @set-period="setPeriod"
            />
          </div>
        </div>
      </Teleport>
    </div>
  </div>
</template>

<style scoped>
@keyframes slide-in-right {
  from { transform: translateX(100%); }
  to { transform: translateX(0); }
}

.animate-slide-in-right {
  animation: slide-in-right 0.25s ease-out;
}

.scrollbar-none {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-none::-webkit-scrollbar {
  display: none;
}
</style>
