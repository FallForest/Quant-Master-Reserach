import { ref, nextTick, onUnmounted } from 'vue'
import { api } from '../utils/api'
import { isMinutePeriod, isDailyLikePeriod, aggregateKline, REALTIME_DAY_REFRESH_MS } from '../utils/kline'
import CandlestickChart from '../charts/CandlestickChart'
import { useManagedInterval } from './useManagedInterval'
import { useMarketStatus } from './useMarketStatus'

export function useKlineDetail(options = {}) {
  const chartId = options.chartId || 'kline-chart'
  const mobileChartId = options.mobileChartId || 'kline-chart-mobile'

  const selectedSymbol = ref(null)
  const selectedName = ref('')
  const showKline = ref(false)
  const period = ref('D')
  const loadingMin = ref(false)
  const quote = ref({ close: 0, change: 0, changePct: '0' })

  let rawDaily = []
  let rawMin1 = []
  let chart = null
  let resizeObserver = null
  let currentAbortController = null

  const detailTimers = useManagedInterval()
  const { marketOpen, checkMarketStatus } = useMarketStatus()

  function updateQuote(q) {
    if (q) quote.value = q
  }

  function findContainer() {
    const s = document.getElementById(chartId)
    if (s?.offsetParent) return s
    const m = document.getElementById(mobileChartId)
    if (m?.clientWidth) return m
    return null
  }

  function applyPeriod() {
    const currentPeriod = period.value
    const isMin = isMinutePeriod(currentPeriod)
    const src = isMin ? rawMin1 : rawDaily
    if (!src.length) return

    const last = src[src.length - 1]
    const prev = src.length > 1 ? src[src.length - 2] : null
    const change = prev ? +(last.close - prev.close).toFixed(2) : 0
    const changePct = prev && prev.close ? +(change / prev.close * 100).toFixed(2) : 0
    quote.value = { ...quote.value, close: last.close, change, changePct: String(changePct), volume: last.volume }

    const data = (currentPeriod === 'D' || currentPeriod === '1min')
      ? src
      : aggregateKline(src, currentPeriod)

    const container = findContainer()
    if (!container) return

    if (chart) {
      chart.setPeriod(currentPeriod)
      chart.setData(data)
      chart.resize()
      return
    }
    if (resizeObserver) resizeObserver.disconnect()
    chart = new CandlestickChart(container)
    chart.setPeriod(currentPeriod)
    chart.setData(data)
    chart.resize()
    resizeObserver = new ResizeObserver(() => chart?.resize())
    resizeObserver.observe(container)
  }

  async function loadDailyKline(sym, signal) {
    return api(`/api/browser/kline/${sym}?freq=1d&includeRealtime=1`, { signal })
  }

  async function selectStock(symbol, name) {
    if (currentAbortController) currentAbortController.abort()
    currentAbortController = new AbortController()
    const signal = currentAbortController.signal

    detailTimers.clearManagedTimers()
    selectedSymbol.value = symbol
    selectedName.value = name || ''
    showKline.value = true

    await nextTick()

    const dayData = await loadDailyKline(symbol, signal)
    if (signal.aborted) return

    if (dayData?.kline?.length) {
      rawDaily = dayData.kline
      rawMin1 = []
      const last = dayData.kline[dayData.kline.length - 1]
      updateQuote({ ...dayData.quote, open: last.open, high: last.high, low: last.low, volume: last.volume })
      if (isMinutePeriod(period.value)) period.value = 'D'
    } else {
      rawDaily = []
      rawMin1 = []
    }

    applyPeriod()

    // Minute polling (1s)
    detailTimers.setManagedInterval(async () => {
      if (signal.aborted || selectedSymbol.value !== symbol || !isMinutePeriod(period.value)) return
      const fresh = await api(`/api/realtime/kline/${symbol}`, { signal })
      if (signal.aborted || selectedSymbol.value !== symbol) return
      if (fresh?.kline?.length) {
        rawMin1 = fresh.kline
        applyPeriod()
      }
    }, 1000)

    // Daily refresh during market hours
    checkMarketStatus()
    detailTimers.setManagedInterval(async () => {
      if (signal.aborted || selectedSymbol.value !== symbol || !isDailyLikePeriod(period.value)) return
      if (!marketOpen.value) return
      const freshDaily = await loadDailyKline(symbol, signal)
      if (signal.aborted || selectedSymbol.value !== symbol) return
      if (freshDaily?.kline?.length) {
        rawDaily = freshDaily.kline
        if (freshDaily?.quote) {
          const last = freshDaily.kline[freshDaily.kline.length - 1]
          updateQuote({ ...freshDaily.quote, open: last.open, high: last.high, low: last.low, volume: last.volume })
        }
        applyPeriod()
      }
    }, REALTIME_DAY_REFRESH_MS)

    // Realtime quote (1s)
    detailTimers.setManagedInterval(async () => {
      if (signal.aborted || selectedSymbol.value !== symbol) return
      checkMarketStatus()
      if (!marketOpen.value) return
      const realtime = await api(`/api/realtime/quote/${symbol}`, { signal })
      if (signal.aborted || selectedSymbol.value !== symbol) return
      if (realtime?.ok && realtime.quote) {
        const t = realtime.quote
        const lastClose = t.lastClose || 0
        quote.value = {
          close: t.price,
          change: lastClose > 0 ? +(t.price - lastClose).toFixed(2) : 0,
          changePct: lastClose > 0 ? String(+((t.price - lastClose) / lastClose * 100).toFixed(2)) : '0',
          open: t.open, high: t.high, low: t.low,
          volume: t.vol * 100, amount: t.amount, lastClose,
        }
      }
    }, 1000)
  }

  async function setPeriod(nextPeriod) {
    period.value = nextPeriod
    const sym = selectedSymbol.value
    if (!sym) return

    if (isMinutePeriod(nextPeriod)) {
      loadingMin.value = true
      const minuteData = await api(`/api/realtime/kline/${sym}`)
      loadingMin.value = false
      if (selectedSymbol.value !== sym) return
      rawMin1 = minuteData?.kline?.length ? minuteData.kline : []
      if (minuteData?.quote) updateQuote(minuteData.quote)
    }
    applyPeriod()
  }

  function closeDetail() {
    if (currentAbortController) {
      currentAbortController.abort()
      currentAbortController = null
    }
    detailTimers.clearManagedTimers()
    showKline.value = false
    selectedSymbol.value = null
    selectedName.value = ''
    rawDaily = []
    rawMin1 = []
    if (chart) { chart.destroy(); chart = null }
    if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null }
  }

  onUnmounted(() => {
    if (currentAbortController) { currentAbortController.abort(); currentAbortController = null }
    detailTimers.clearManagedTimers()
    showKline.value = false
    if (chart) { chart.destroy(); chart = null }
    if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null }
  })

  return {
    selectedSymbol,
    selectedName,
    showKline,
    period,
    loadingMin,
    quote,
    marketOpen,
    checkMarketStatus,
    selectStock,
    setPeriod,
    closeDetail,
    applyPeriod,
  }
}
