<script setup>
import { computed, onActivated, onDeactivated, onMounted, onUnmounted, nextTick, ref, watch } from 'vue'
import { api, fmtNum } from '../utils/api'
import * as echarts from 'echarts'
import CandlestickChart from '../charts/CandlestickChart'
import KlineContent from '../components/KlineContent.vue'
import { isMinutePeriod, isDailyLikePeriod, aggregateKline, REALTIME_DAY_REFRESH_MS } from '../utils/kline'
import { useMarketStatus } from '../composables/useMarketStatus'

// ---- 状态 ----
const loading = ref(true)
const loadingPredictions = ref(false)
const models = ref([])
const selectedModel = ref('')
const modelInfo = ref(null)

// 预测相关
const availableDates = ref([])
const selectedDate = ref('')
const topK = ref(20)
const predictions = ref(null)
const selectedStock = ref(null)
const pipelineStatus = ref(null)

// 实时选股
const liveDate = ref('')
const liveRunning = ref(false)
const liveError = ref('')
const liveResult = ref(null)

// 自选
const watchlistSymbols = ref([])
const watchlistLoading = ref(false)
const watchlistError = ref('')
const watchlistPending = ref({})
const watchlistSet = computed(() => new Set(watchlistSymbols.value))

// 右侧详情 / K线
const showDetailPanel = ref(false)
const detailLoading = ref(false)
const loadingMin = ref(false)
const period = ref('D')
const { marketOpen, checkMarketStatus } = useMarketStatus()
const quote = ref(createEmptyQuote())

// 指标
const metrics = ref([
  { label: '年化超额收益', value: '--', desc: '无成本年化超额', color: 'text-bull' },
  { label: '信息比率', value: '--', desc: '超额收益 IR', color: 'text-brand-600' },
  { label: '最大回撤', value: '--', desc: '无成本最大回撤', color: 'text-danger' },
  { label: 'ICIR', value: '--', desc: 'IC 信息比率', color: 'text-brand-600' },
  { label: 'Rank ICIR', value: '--', desc: '排序 IC 信息比率', color: 'text-brand-600' },
  { label: '选股数', value: '--', desc: '每日持仓股票数', color: 'text-success' },
])

// 图表实例
let scoreChart = null
let detailChart = null
let resizeObserver = null
let currentAbortController = null
let detailPollingStarted = false
let pageActive = false
let rawDaily = []
let rawMin1 = []

const detailTimers = new Set()

async function loadDailyKline(instrument, signal) {
  return api(`/api/browser/kline/${instrument}?freq=1d&includeRealtime=1`, { signal })
}

const hasSelectedStock = computed(() => showDetailPanel.value && !!selectedStock.value)
const selectedStockRank = computed(() => selectedStock.value?.rank ?? '--')
const selectedScorePercent = computed(() => {
  if (!selectedStock.value || !predictions.value?.scoreStats) return '--'
  return `${scorePercent(selectedStock.value.score, predictions.value.scoreStats).toFixed(1)}%`
})
const selectedScoreRange = computed(() => {
  if (!predictions.value?.scoreStats) return '--'
  return `${predictions.value.scoreStats.min} ~ ${predictions.value.scoreStats.max}`
})
const selectedStockWatchlisted = computed(() => {
  const instrument = selectedStock.value?.instrument
  return instrument ? watchlistSet.value.has(instrument) : false
})
const rankingColClass = computed(() => (hasSelectedStock.value ? 'lg:col-span-2' : 'lg:col-span-5'))

function createEmptyQuote() {
  return {
    close: null,
    change: 0,
    changePct: '0',
    open: null,
    high: null,
    low: null,
    volume: null,
    amount: null,
    lastClose: null,
  }
}

function setManagedInterval(timerSet, fn, ms) {
  const id = setInterval(fn, ms)
  timerSet.add(id)
  return id
}

function clearManagedTimers(timerSet) {
  for (const id of timerSet) clearInterval(id)
  timerSet.clear()
}

function cleanupDetailChart() {
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  if (detailChart) {
    detailChart.destroy()
    detailChart = null
  }
}

function stopDetailPolling() {
  clearManagedTimers(detailTimers)
  detailPollingStarted = false
}

function resetDetailState() {
  stopDetailPolling()
  cleanupDetailChart()
  rawDaily = []
  rawMin1 = []
  quote.value = createEmptyQuote()
  loadingMin.value = false
  detailLoading.value = false
  period.value = 'D'
  if (currentAbortController) {
    currentAbortController.abort()
    currentAbortController = null
  }
}

function handleResize() {
  scoreChart?.resize()
  detailChart?.resize()
}

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  pageActive = true
  loading.value = true
  const today = new Date()
  liveDate.value = today.toISOString().slice(0, 10)

  await loadWatchlist()
  await loadPipelineStatus()

  const modelData = await api('/api/models')
  if (modelData?.models?.length) {
    models.value = modelData.models
    selectedModel.value = modelData.models[0].alias
    await loadModelData()
  }
  loading.value = false
})

onActivated(() => {
  pageActive = true
  if (selectedStock.value) {
    void refreshSelectedStockDetail(selectedStock.value)
  }
})

onDeactivated(() => {
  pageActive = false
  stopDetailPolling()
  cleanupDetailChart()
  if (currentAbortController) {
    currentAbortController.abort()
    currentAbortController = null
  }
})

onUnmounted(() => {
  pageActive = false
  scoreChart?.dispose()
  resetDetailState()
  window.removeEventListener('resize', handleResize)
})

watch(selectedModel, async () => {
  if (!selectedModel.value) return
  loading.value = true
  closeSelectedStockPanel(false)
  await loadModelData()
  loading.value = false
})

watch(selectedDate, async () => {
  if (!selectedDate.value || !selectedModel.value) return
  await loadPredictions()
})

watch(topK, async () => {
  if (!selectedDate.value || !selectedModel.value) return
  await loadPredictions()
})

function mergeAvailableDates(dates = [], extraDate = '') {
  const merged = Array.from(new Set([...(dates || []), ...(extraDate ? [extraDate] : [])]))
  merged.sort()
  return merged
}

function predictionDisplayDate(data = predictions.value) {
  if (!data) return '--'
  return data.requestedDate || data.date || selectedDate.value || '--'
}

function predictionFeatureDate(data = predictions.value) {
  if (!data) return '--'
  return data.featureDate || data.date || selectedDate.value || '--'
}

function latestMarketDate() {
  return pipelineStatus.value?.marketEffectiveLastDate || '--'
}

function latestCalendarDate() {
  return pipelineStatus.value?.calendarLastDate || '--'
}

function liveStatusHint() {
  if (!pipelineStatus.value) return '将基于最新已落盘交易日数据生成候选排名'
  if (pipelineStatus.value.syncing) return '数据同步进行中，请等待同步完成后再运行下一交易日选股'
  if (pipelineStatus.value.syncError) return `同步状态异常：${pipelineStatus.value.syncError}`
  return `最新市场数据日 ${latestMarketDate()}，日历最新日 ${latestCalendarDate()}`
}

async function loadPipelineStatus() {
  const data = await api('/api/pipeline/status')
  if (data && !data.error) pipelineStatus.value = data
}

async function loadModelData() {
  const alias = selectedModel.value
  const [infoData, datesData] = await Promise.all([
    api(`/api/models/${alias}/info`),
    api(`/api/models/${alias}/dates`),
  ])

  if (infoData) {
    modelInfo.value = infoData
    applyMetrics(infoData)
  }

  if (datesData?.dates?.length) {
    availableDates.value = datesData.dates
    selectedDate.value = datesData.dates[datesData.dates.length - 1]
  }
}

function syncSelectedStockFromPredictions() {
  if (!selectedStock.value || !predictions.value?.stocks?.length) return
  const nextStock = predictions.value.stocks.find((stock) => stock.instrument === selectedStock.value.instrument)
  if (!nextStock) {
    closeSelectedStockPanel(false)
    return
  }
  selectedStock.value = nextStock
}

async function loadPredictions() {
  if (!selectedModel.value || !selectedDate.value) return
  loadingPredictions.value = true

  const data = await api(`/api/models/${selectedModel.value}/predictions?date=${selectedDate.value}&top_k=${topK.value}`)
  if (data) {
    predictions.value = data
    syncSelectedStockFromPredictions()
    await nextTick()
    renderScoreChart()
    if (selectedStock.value && pageActive) {
      await refreshSelectedStockDetail(selectedStock.value)
    }
  }
  loadingPredictions.value = false
}

function applyMetrics(info) {
  const m = info.metrics || {}
  metrics.value[0].value = m.annualizedReturn != null ? `${(m.annualizedReturn * 100).toFixed(2)}%` : '--'
  metrics.value[1].value = m.informationRatio != null ? m.informationRatio.toFixed(4) : '--'
  metrics.value[2].value = m.maxDrawdown != null ? `${(m.maxDrawdown * 100).toFixed(2)}%` : '--'
  metrics.value[3].value = m.icir != null ? m.icir.toFixed(4) : '--'
  metrics.value[4].value = m.rankIcir != null ? m.rankIcir.toFixed(4) : '--'
  metrics.value[5].value = info.params?.topk ?? '--'
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

function setWatchlistPending(symbol, pending) {
  watchlistPending.value = {
    ...watchlistPending.value,
    [symbol]: pending,
  }
}

function isWatchlisted(symbol) {
  return watchlistSet.value.has(symbol)
}

async function toggleSelectedWatchlist() {
  const symbol = selectedStock.value?.instrument
  if (!symbol || watchlistPending.value[symbol]) return

  setWatchlistPending(symbol, true)
  watchlistError.value = ''
  try {
    const data = isWatchlisted(symbol)
      ? await api(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' })
      : await api('/api/watchlist', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol }),
        })
    watchlistSymbols.value = Array.isArray(data?.symbols) ? data.symbols : watchlistSymbols.value
  } catch {
    watchlistError.value = isWatchlisted(symbol) ? '移出自选失败' : '加入自选失败'
  } finally {
    setWatchlistPending(symbol, false)
  }
}

function normalizeQuoteData(instrument, data) {
  const stockQuote = data?.quotes?.[instrument] || data?.quotes?.[instrument.toUpperCase?.()] || data?.quotes?.[instrument.replace(/^SH|^SZ|^BJ/, '')]
  if (!stockQuote) return null
  return {
    close: stockQuote.close ?? null,
    change: stockQuote.change ?? 0,
    changePct: String(stockQuote.changePct ?? 0),
    amount: stockQuote.amount ?? null,
    volume: stockQuote.volume ?? null,
  }
}

function mergeQuoteData(nextQuote = {}) {
  quote.value = {
    ...quote.value,
    ...nextQuote,
  }
}

async function loadSelectedStockQuote(instrument, signal) {
  const data = await api(`/api/browser/quotes?symbols=${instrument}`, { signal })
  const normalized = normalizeQuoteData(instrument, data)
  if (normalized) mergeQuoteData(normalized)
}

function updateQuote(nextQuote) {
  if (!nextQuote) return
  mergeQuoteData({
    close: nextQuote.close ?? quote.value.close,
    change: nextQuote.change ?? quote.value.change,
    changePct: String(nextQuote.changePct ?? quote.value.changePct ?? '0'),
  })
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
    mergeQuoteData({
      close: last.close,
      change,
      changePct: String(changePct),
      lastClose: prev?.close ?? quote.value.lastClose,
    })
  }

  const data = (currentPeriod === 'D' || currentPeriod === '1min')
    ? source
    : aggregateKline(source, currentPeriod)
  const container = document.getElementById('kline-chart')
  if (!container) return
  // 轮询更新时复用已有 chart，避免 DOM 重建导致闪烁和缩放丢失
  if (detailChart) {
    detailChart.setPeriod(currentPeriod)
    detailChart.setData(data)
    return
  }
  cleanupDetailChart()
  detailChart = new CandlestickChart(container)
  detailChart.setPeriod(currentPeriod)
  detailChart.setData(data)

  resizeObserver = new ResizeObserver(() => {
    if (detailChart) detailChart.resize()
  })
  resizeObserver.observe(container)
}

async function setPeriod(nextPeriod) {
  period.value = nextPeriod
  const isMinute = isMinutePeriod(nextPeriod)

  if (isMinute && selectedStock.value) {
    const symbol = selectedStock.value.instrument
    loadingMin.value = true
    const minuteData = await api(`/api/realtime/kline/${symbol}`)
    loadingMin.value = false
    if (selectedStock.value?.instrument !== symbol) return
    rawMin1 = minuteData?.kline?.length ? minuteData.kline : []
    if (minuteData?.quote) updateQuote(minuteData.quote)
  }

  applyPeriod()
}

async function refreshSelectedStockDetail(stock) {
  if (!stock?.instrument || !pageActive) return

  resetDetailState()
  selectedStock.value = stock
  showDetailPanel.value = true
  detailLoading.value = true
  quote.value = createEmptyQuote()
  checkMarketStatus()

  currentAbortController = new AbortController()
  const signal = currentAbortController.signal

  await nextTick()

  const instrument = stock.instrument
  const [dayData] = await Promise.all([
    loadDailyKline(instrument, signal),
    loadSelectedStockQuote(instrument, signal),
  ])

  if (signal.aborted || selectedStock.value?.instrument !== instrument) return

  if (dayData?.kline?.length) {
    rawDaily = dayData.kline
    rawMin1 = []
    updateQuote(dayData.quote)
  } else {
    rawDaily = []
    rawMin1 = []
  }

  detailLoading.value = false
  await nextTick()
  if (signal.aborted || selectedStock.value?.instrument !== instrument) return
  applyPeriod()
  startDetailPolling(instrument, signal)
}

function startDetailPolling(instrument, signal) {
  if (detailPollingStarted) stopDetailPolling()
  detailPollingStarted = true

  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedStock.value?.instrument !== instrument || !isDailyLikePeriod(period.value)) return
    const freshDaily = await loadDailyKline(instrument, signal)
    if (signal.aborted || selectedStock.value?.instrument !== instrument) return
    if (freshDaily?.kline?.length) {
      rawDaily = freshDaily.kline
      if (freshDaily?.quote) updateQuote(freshDaily.quote)
      applyPeriod()
    }
  }, REALTIME_DAY_REFRESH_MS)

  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedStock.value?.instrument !== instrument) return
    const quoteData = await api(`/api/realtime/quote/${instrument}`, { signal })
    if (signal.aborted || selectedStock.value?.instrument !== instrument) return
    if (quoteData?.ok && quoteData.quote) {
      const tick = quoteData.quote
      const lastClose = tick.lastClose || 0
      const change = lastClose > 0 ? +(tick.price - lastClose).toFixed(2) : 0
      const changePct = lastClose > 0 ? +(change / lastClose * 100).toFixed(2) : 0
      mergeQuoteData({
        close: tick.price,
        change,
        changePct: String(changePct),
        open: tick.open,
        high: tick.high,
        low: tick.low,
        volume: tick.vol,
        amount: tick.amount,
        lastClose,
      })
    }
    checkMarketStatus()
  }, 1000)

  setManagedInterval(detailTimers, async () => {
    if (signal.aborted || selectedStock.value?.instrument !== instrument || !isMinutePeriod(period.value)) return
    const minuteData = await api(`/api/realtime/kline/${instrument}`, { signal })
    if (signal.aborted || selectedStock.value?.instrument !== instrument) return
    if (minuteData?.kline?.length) {
      rawMin1 = minuteData.kline
      if (minuteData?.quote) updateQuote(minuteData.quote)
      applyPeriod()
    }
  }, 1000)
}

async function closeSelectedStockPanel(resetSelection = true) {
  resetDetailState()
  showDetailPanel.value = false
  if (resetSelection) selectedStock.value = null
}

async function selectStock(stock) {
  if (!stock?.instrument) return
  if (selectedStock.value?.instrument === stock.instrument && showDetailPanel.value) {
    await closeSelectedStockPanel(true)
    return
  }
  await refreshSelectedStockDetail(stock)
}

async function runLivePrediction() {
  if (!liveDate.value || !selectedModel.value) return
  liveRunning.value = true
  liveError.value = ''
  liveResult.value = null

  try {
    const data = await api(`/api/models/${selectedModel.value}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date: liveDate.value, top_k: topK.value }),
    })
    if (data?.error) {
      liveError.value = data.error
    } else if (data?.stocks) {
      liveResult.value = data
      predictions.value = data
      await loadPipelineStatus()
      const datesData = await api(`/api/models/${selectedModel.value}/dates`)
      availableDates.value = mergeAvailableDates(
        [...(datesData?.dates || []), ...availableDates.value],
        data.requestedDate || data.featureDate,
      )
      selectedDate.value = data.requestedDate || data.featureDate || data.date
      await nextTick()
      renderScoreChart()
      syncSelectedStockFromPredictions()
      if (selectedStock.value && pageActive) {
        await refreshSelectedStockDetail(selectedStock.value)
      }
    }
  } catch (e) {
    liveError.value = e.message || '运行失败'
  } finally {
    liveRunning.value = false
  }
}

function renderScoreChart() {
  const el = document.getElementById('score-dist-chart')
  if (!el || !predictions.value?.stocks?.length) return
  scoreChart?.dispose()
  scoreChart = echarts.init(el)

  const stocks = predictions.value.stocks
  const instruments = stocks.map((stock) => stock.instrument.replace(/^(SH|SZ)/, ''))
  const scores = stocks.map((stock) => stock.score)

  scoreChart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15,23,42,0.92)',
      borderColor: '#1E40AF',
      textStyle: { color: '#F8FAFC', fontFamily: 'Fira Code, monospace', fontSize: 12 },
      formatter(params) {
        const point = params[0]
        const stock = stocks[point.dataIndex]
        return `<div style="font-weight:600">${stock.instrument}</div>` +
          `<div style="color:#64748B;margin-top:2px">${stock.name || '未知名称'}</div>` +
          `<div>排名: #${stock.rank}</div>` +
          `<div>分数: <b style="color:${point.color}">${stock.score.toFixed(4)}</b></div>`
      },
    },
    grid: { left: 50, right: 20, top: 20, bottom: 50 },
    xAxis: {
      type: 'category',
      data: instruments,
      axisLabel: { color: '#64748B', fontSize: 9, rotate: 45, fontFamily: 'Fira Code' },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '预测分数',
      nameTextStyle: { color: '#94A3B8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#F1F5F9', type: 'dashed' } },
      axisLabel: { color: '#94A3B8', fontSize: 10, fontFamily: 'Fira Code' },
    },
    series: [{
      type: 'bar',
      data: scores.map((score) => ({
        value: score,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: score >= 0 ? 'rgba(239,68,68,0.8)' : 'rgba(16,185,129,0.8)' },
            { offset: 1, color: score >= 0 ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)' },
          ]),
          borderRadius: [3, 3, 0, 0],
        },
      })),
      barMaxWidth: 20,
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.2)' },
      },
    }],
  })
}

function scorePercent(score, stats) {
  if (!stats || stats.max === stats.min) return 50
  return Math.max(5, Math.min(95, ((score - stats.min) / (stats.max - stats.min)) * 100))
}

function quoteToneClass(value) {
  if (value > 0) return 'text-bull'
  if (value < 0) return 'text-bear'
  return 'text-slate-500'
}

function quoteBadgeClass(value) {
  if (value > 0) return 'bg-red-50 text-red-600 border-red-100'
  if (value < 0) return 'bg-emerald-50 text-emerald-600 border-emerald-100'
  return 'bg-surface-2 text-slate-500 border-surface-3'
}

function formatSigned(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return '--'
  const numeric = Number(value)
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(digits)}`
}

function formatAmount(value) {
  if (value == null || Number.isNaN(Number(value))) return '--'
  return fmtNum(Number(value))
}

function formatDate(value) {
  return value || '--'
}

function currentModel() {
  return models.value.find((model) => model.alias === selectedModel.value) || {}
}
</script>

<template>
  <div class="h-full p-4 sm:p-6 space-y-5 animate-slide-in overflow-y-auto">
    <div class="flex flex-wrap items-center gap-3">
      <h2 class="text-base font-semibold text-slate-700">模型选股</h2>
      <div class="flex-1"></div>
      <select
        v-model="selectedModel"
        class="px-3 py-1.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer"
      >
        <option v-for="m in models" :key="m.alias" :value="m.alias">
          {{ m.name || m.alias }}
        </option>
      </select>
    </div>
    <p v-if="currentModel().description" class="text-xs text-slate-500 -mt-3">
      {{ currentModel().description }}
    </p>

    <div class="bg-white rounded-xl border border-surface-3 p-4">
      <div class="flex flex-wrap items-center gap-3">
        <div class="flex items-center gap-2">
          <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 010 1.972l-11.54 6.347a1.125 1.125 0 01-1.667-.986V5.653z"/>
          </svg>
          <h3 class="text-sm font-semibold text-slate-600">收盘后生成下一交易日选股</h3>
        </div>
        <span class="text-[10px] text-slate-400">基于最新已落盘交易日数据，为目标交易日生成候选排名</span>
        <div class="flex-1"></div>
        <div class="flex items-center gap-2">
          <label class="text-xs text-slate-500">目标日期</label>
          <input
            v-model="liveDate"
            type="date"
            class="px-2.5 py-1.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer font-mono"
          />
          <button
            :disabled="liveRunning || !liveDate"
            :class="[
              'flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition cursor-pointer',
              liveRunning || !liveDate
                ? 'bg-surface-2 text-slate-400 cursor-not-allowed'
                : 'bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800',
            ]"
            @click="runLivePrediction"
          >
            <svg v-if="liveRunning" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 010 1.972l-11.54 6.347a1.125 1.125 0 01-1.667-.986V5.653z"/>
            </svg>
            {{ liveRunning ? '运行中...' : '运行选股' }}
          </button>
        </div>
      </div>

      <div class="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-500 bg-slate-50 rounded-lg px-3 py-2">
        <span>最新市场数据日 {{ latestMarketDate() }}</span>
        <span>·</span>
        <span>日历最新日 {{ latestCalendarDate() }}</span>
        <span v-if="pipelineStatus?.syncing">· 数据同步中</span>
        <span v-if="pipelineStatus?.syncError" class="text-danger">· {{ pipelineStatus.syncError }}</span>
        <span v-if="!pipelineStatus" class="text-slate-400">· {{ liveStatusHint() }}</span>
      </div>
      <!-- 同步进度条 -->
      <div v-if="pipelineStatus?.syncing && pipelineStatus?.syncProgress" class="mt-2 flex items-center gap-3 text-xs bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
        <svg class="w-3.5 h-3.5 animate-spin text-amber-500" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        <span class="text-amber-700">{{ pipelineStatus.syncProgress.label || '数据同步中...' }}</span>
        <div v-if="pipelineStatus.syncProgress.total > 0" class="flex-1 max-w-[160px] bg-amber-200 rounded-full h-1.5">
          <div
            class="bg-amber-500 rounded-full h-1.5 transition-all duration-500"
            :style="{ width: (pipelineStatus.syncProgress.done / pipelineStatus.syncProgress.total * 100) + '%' }"
          ></div>
        </div>
        <span v-if="pipelineStatus.syncProgress.total > 0" class="text-amber-600 font-mono tabular-nums">{{ pipelineStatus.syncProgress.done }}/{{ pipelineStatus.syncProgress.total }}</span>
      </div>

      <div v-if="liveError" class="mt-3 flex items-center gap-2 text-xs text-danger bg-danger/5 rounded-lg px-3 py-2">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
        </svg>
        {{ liveError }}
      </div>

      <div v-if="liveResult && !liveError" class="mt-3 flex items-center gap-2 text-xs text-success bg-success/5 rounded-lg px-3 py-2">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <span v-if="liveResult.dateMapped">
          下一交易日候选已生成 · 目标交易日 {{ predictionDisplayDate(liveResult) }} · 使用 {{ predictionFeatureDate(liveResult) }} 收盘数据 · 共 {{ liveResult.totalStocks }} 只股票
        </span>
        <span v-else>
          选股结果已生成 · 交易日 {{ predictionDisplayDate(liveResult) }} · 共 {{ liveResult.totalStocks }} 只股票
        </span>
      </div>
    </div>

    <div v-if="loading" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <div v-for="i in 6" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
        <div class="skeleton h-3 w-16 mb-2"></div>
        <div class="skeleton h-6 w-20 mb-1"></div>
        <div class="skeleton h-2.5 w-24"></div>
      </div>
    </div>
    <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <div v-for="m in metrics" :key="m.label" class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
        <div class="text-[11px] text-slate-500 mb-1">{{ m.label }}</div>
        <div :class="['text-xl font-bold font-mono', m.color]">{{ m.value }}</div>
        <div class="text-[10px] text-slate-500 mt-0.5">{{ m.desc }}</div>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <label class="text-xs text-slate-500">数据日期</label>
        <select
          v-model="selectedDate"
          class="px-2.5 py-1.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer"
        >
          <option v-for="d in availableDates" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>
      <div class="flex items-center gap-2">
        <label class="text-xs text-slate-500">Top K</label>
        <select
          v-model.number="topK"
          class="px-2.5 py-1.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer"
        >
          <option :value="10">10</option>
          <option :value="20">20</option>
          <option :value="30">30</option>
          <option :value="50">50</option>
        </select>
      </div>
      <div class="flex-1"></div>
      <div v-if="predictions" class="text-xs text-slate-500">
        共 {{ predictions.totalStocks }} 只 · 分数范围 [{{ predictions.scoreStats?.min }}, {{ predictions.scoreStats?.max }}]
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-5 gap-4">
      <div :class="[rankingColClass, 'bg-white rounded-xl border border-surface-3 p-4 transition-all duration-300']">
        <h3 class="text-sm font-semibold text-slate-600 mb-3">
          选股排名
          <span
            v-if="predictions?.source === 'live'"
            class="ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full bg-brand-50 text-brand-600 font-medium"
          >
            实时
          </span>
          <span class="text-[10px] font-normal text-slate-400 ml-1">点击行查看右侧详情</span>
        </h3>

        <div v-if="loadingPredictions" class="space-y-2">
          <div v-for="i in 8" :key="i" class="skeleton h-9 w-full rounded-lg"></div>
        </div>
        <div v-else-if="!predictions?.stocks?.length" class="text-center text-sm text-slate-400 py-8">
          暂无数据
        </div>
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                <th class="py-2 pr-2 w-12">排名</th>
                <th class="py-2 pr-2">代码 / 名称</th>
                <th class="py-2 pr-2 text-right">分数</th>
                <th class="py-2 pl-2 w-32">分数分布</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="s in predictions.stocks"
                :key="s.instrument"
                :class="[
                  'stock-row cursor-pointer transition border-b border-surface-3/50 last:border-0',
                  selectedStock?.instrument === s.instrument && hasSelectedStock
                    ? 'bg-brand-50/70'
                    : 'hover:bg-surface-2/60',
                ]"
                @click="selectStock(s)"
              >
                <td class="py-2.5 pr-2 font-mono text-xs align-top">
                  <span
                    :class="[
                      'inline-flex items-center justify-center w-6 h-6 rounded-md text-[10px] font-bold',
                      s.rank <= 3 ? 'bg-cta/10 text-cta' : 'bg-surface-2 text-slate-500',
                    ]"
                  >{{ s.rank }}</span>
                </td>
                <td class="py-2.5 pr-2 align-top">
                  <div class="flex items-start justify-between gap-3 min-w-0">
                    <div class="flex flex-col min-w-0">
                      <span class="font-mono text-xs text-slate-700">{{ s.instrument }}</span>
                      <span class="text-[11px] text-slate-500 truncate">{{ s.name || '未知名称' }}</span>
                    </div>
                    <span
                      :class="[
                        'mt-0.5 inline-flex h-6 min-w-6 px-1 flex-shrink-0 items-center justify-center rounded-full border text-[10px] transition-colors',
                        selectedStock?.instrument === s.instrument && hasSelectedStock
                          ? 'border-brand-200 bg-brand-50 text-brand-600'
                          : 'border-surface-3 bg-white text-slate-400',
                      ]"
                    >
                      {{ selectedStock?.instrument === s.instrument && hasSelectedStock ? '已选' : '查看' }}
                    </span>
                  </div>
                </td>
                <td class="py-2.5 pr-2 text-right font-mono text-xs align-top" :class="s.score >= 0 ? 'text-bull' : 'text-bear'">
                  {{ s.score.toFixed(4) }}
                </td>
                <td class="py-2.5 pl-2 align-top">
                  <div class="w-full bg-surface-2 rounded-full h-2 mt-1">
                    <div
                      class="h-2 rounded-full transition-all duration-300"
                      :class="s.score >= 0 ? 'bg-bull/70' : 'bg-bear/70'"
                      :style="{ width: scorePercent(s.score, predictions.scoreStats) + '%' }"
                    ></div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="hasSelectedStock" class="lg:col-span-3 bg-white rounded-xl border border-surface-3 p-4 flex flex-col gap-4 lg:sticky lg:top-4 self-start lg:max-h-[calc(100vh-7rem)] overflow-x-hidden overflow-y-visible lg:overflow-y-auto">
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div class="flex flex-wrap items-center gap-2">
              <h3 class="text-base font-semibold text-slate-700">{{ selectedStock.name || selectedStock.instrument }}</h3>
              <span class="font-mono text-xs text-slate-500">{{ selectedStock.instrument }}</span>
            </div>
            <div class="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
              <span class="inline-flex items-center rounded-full bg-brand-50 px-2 py-0.5 text-brand-600">当前排名 #{{ selectedStockRank }}</span>
              <span class="inline-flex items-center rounded-full bg-surface-2 px-2 py-0.5 text-slate-500">
                {{ predictions?.source === 'live' ? '收盘后选股' : '历史预测' }}
              </span>
              <span v-if="watchlistError" class="text-danger">{{ watchlistError }}</span>
            </div>
          </div>
          <div class="flex items-center gap-2 self-start">
            <button
              type="button"
              :disabled="watchlistPending[selectedStock?.instrument]"
              :class="[
                'inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors cursor-pointer',
                selectedStockWatchlisted
                  ? 'border-amber-200 bg-amber-50 text-amber-600 hover:bg-amber-100'
                  : 'border-surface-3 bg-white text-slate-600 hover:border-amber-200 hover:bg-amber-50 hover:text-amber-600',
                watchlistPending[selectedStock?.instrument] ? 'opacity-60 cursor-wait' : '',
              ]"
              @click="toggleSelectedWatchlist"
            >
              <svg class="h-3.5 w-3.5" :fill="selectedStockWatchlisted ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321 1.01l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.386a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.56a.562.562 0 01-.84-.61l1.285-5.386a.563.563 0 00-.182-.557L3.04 10.407a.563.563 0 01.321-1.01l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/>
              </svg>
              {{ selectedStockWatchlisted ? '已在自选' : '加入自选' }}
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <div class="rounded-lg bg-surface-1 px-3 py-3">
            <div class="text-[11px] text-slate-500">交易日</div>
            <div class="mt-1 font-mono text-sm text-slate-700">{{ predictionDisplayDate() }}</div>
          </div>
          <div class="rounded-lg bg-surface-1 px-3 py-3">
            <div class="text-[11px] text-slate-500">数据日</div>
            <div class="mt-1 font-mono text-sm text-slate-700">{{ predictionFeatureDate() }}</div>
          </div>
          <div class="rounded-lg bg-surface-1 px-3 py-3">
            <div class="text-[11px] text-slate-500">当前分数</div>
            <div :class="['mt-1 font-mono text-sm font-semibold', selectedStock.score >= 0 ? 'text-bull' : 'text-bear']">
              {{ selectedStock.score.toFixed(4) }}
            </div>
          </div>
          <div class="rounded-lg bg-surface-1 px-3 py-3">
            <div class="text-[11px] text-slate-500">分数位置</div>
            <div class="mt-1 font-mono text-sm text-slate-700">{{ selectedScorePercent }}</div>
          </div>
          <div class="rounded-lg bg-surface-1 px-3 py-3 xl:col-span-2">
            <div class="text-[11px] text-slate-500">分数区间</div>
            <div class="mt-1 font-mono text-sm text-slate-700">{{ selectedScoreRange }}</div>
          </div>
        </div>

        <div class="rounded-xl border border-surface-3 bg-white overflow-hidden min-h-[420px] flex-1 shrink-0">
          <div v-if="detailLoading" class="h-[420px] p-4 space-y-3">
            <div class="skeleton h-10 w-full rounded-lg"></div>
            <div class="skeleton h-8 w-full rounded-lg"></div>
            <div class="skeleton h-72 w-full rounded-lg"></div>
          </div>
          <KlineContent
            v-else
            :symbol="selectedStock.instrument"
            :name="selectedStock.name"
            :quote="quote"
            :period="period"
            :loadingMin="loadingMin"
            :marketOpen="marketOpen"
            @close="closeSelectedStockPanel"
            @set-period="setPeriod"
          />
        </div>
      </div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-4">
      <h3 class="text-sm font-semibold text-slate-600 mb-3">
        Top {{ topK }} 预测分数分布 <span class="text-[10px] font-normal text-slate-400">数据日 {{ formatDate(selectedDate) }}</span>
      </h3>
      <div v-if="loadingPredictions" class="h-[260px]">
        <div class="skeleton w-full h-full rounded-lg"></div>
      </div>
      <div v-else id="score-dist-chart" class="w-full h-[260px]"></div>
    </div>
  </div>
</template>
