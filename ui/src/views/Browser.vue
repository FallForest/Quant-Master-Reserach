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

// K 线报价
const quote = ref({ close: 0, change: 0, changePct: '0' })

let chart = null
let rawDaily = []
let rawMin1 = []
let refreshTimer = null

const minutePeriods = ['1min']

// ---- 筛选 & 分页 ----
const filtered = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return allStocks.value
  return allStocks.value.filter(s =>
    s.symbol.toLowerCase().includes(q) || (s.name && s.name.toLowerCase().includes(q))
  )
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
onMounted(async () => {
  const data = await api('/api/browser/stocks')
  if (data?.stocks) {
    allStocks.value = data.stocks
  }
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
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
  chart = new CandlestickChart(container)
  chart.setPeriod(p)
  chart.setData(data)
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
  if (chart) { chart.destroy(); chart = null }
}

async function setPeriod(p) {
  period.value = p
  const isMinute = minutePeriods.includes(p)

  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null }

  if (isMinute && selectedSymbol.value) {
    const sym = selectedSymbol.value
    loadingMin.value = true
    const minData = await api(`/api/realtime/kline/${sym}`)
    loadingMin.value = false
    if (currentSymbol !== sym) return
    rawMin1 = minData?.kline?.length ? minData.kline : []
    if (minData?.quote) updateQuote(minData.quote)

    refreshTimer = setInterval(async () => {
      if (currentSymbol !== sym) return
      const fresh = await api(`/api/realtime/kline/${sym}`)
      if (currentSymbol !== sym) return
      if (fresh?.kline?.length) {
        rawMin1 = fresh.kline
        if (fresh.quote) updateQuote(fresh.quote)
        applyPeriod()
      }
    }, 3000)
  }

  applyPeriod()
}

// 工具函数
function chgColor(val) { return val > 0 ? 'text-bull' : val < 0 ? 'text-bear' : 'text-slate-400' }
function chgBg(val) { return val > 0 ? 'bg-red-50' : val < 0 ? 'bg-green-50' : '' }
</script>

<template>
  <div class="h-full flex flex-col animate-slide-in">

    <!-- 工具栏 -->
    <div class="flex-shrink-0 px-4 py-2.5 bg-white border-b border-surface-3 flex items-center gap-3">
      <div class="relative w-64">
        <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none"
             stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        <input v-model="searchQuery" type="text" placeholder="代码或名称..."
          class="w-full pl-9 pr-3 py-1.5 text-sm rounded-lg border border-surface-3
                 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition">
      </div>
      <span class="text-xs text-slate-400">{{ filtered.length }} 只股票</span>
      <div class="flex-1"></div>
      <span v-if="showKline" class="text-xs text-slate-400">
        {{ period === '1min' ? '实时行情 · 3s 刷新' : '历史日线' }}
      </span>
    </div>

    <!-- 主体 -->
    <div class="flex-1 flex min-h-0">

      <!-- 股票列表 -->
      <div :class="['flex flex-col min-h-0 transition-all duration-300',
                     showKline ? 'w-[420px] flex-shrink-0' : 'flex-1']">
        <div class="flex-1 overflow-auto">
          <table class="w-full text-sm">
            <thead class="sticky top-0 bg-white z-10">
              <tr class="text-left text-xs text-slate-400 border-b border-surface-3">
                <th class="px-3 py-2 w-[100px]">代码</th>
                <th class="px-3 py-2">名称</th>
                <th class="px-3 py-2 text-right">最新价</th>
                <th class="px-3 py-2 text-right w-[90px]">涨跌幅</th>
                <th class="px-3 py-2 text-right hidden xl:table-cell">成交量</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="!paged.length">
                <td colspan="5" class="py-12 text-center text-slate-400">
                  {{ searchQuery ? '无匹配结果' : '加载中...' }}
                </td>
              </tr>
              <tr v-for="s in paged" :key="s.symbol"
                  @click="selectStock(s.symbol, s.name)"
                  :class="['cursor-pointer border-b border-surface-100 hover:bg-brand-50/40 transition-colors duration-150',
                           selectedSymbol === s.symbol ? 'bg-brand-50 border-l-2 border-l-brand-500' : 'border-l-2 border-l-transparent']">
                <td class="px-3 py-2 font-mono text-brand-600 text-xs font-medium">{{ s.symbol }}</td>
                <td class="px-3 py-2 text-slate-700 truncate max-w-[120px]">{{ s.name || '--' }}</td>
                <td :class="['px-3 py-2 text-right font-mono font-medium text-sm', chgColor(s.change)]">
                  {{ s.close != null ? s.close.toFixed(2) : '--' }}
                </td>
                <td :class="['px-3 py-2 text-right font-mono text-xs', chgColor(s.change)]">
                  <span :class="['inline-block px-1.5 py-0.5 rounded', chgBg(s.change)]">
                    {{ s.change > 0 ? '+' : '' }}{{ s.changePct != null ? s.changePct + '%' : '--' }}
                  </span>
                </td>
                <td class="px-3 py-2 text-right font-mono text-xs text-slate-400 hidden xl:table-cell">
                  {{ fmtNum(s.volume) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- 分页 -->
        <div class="flex-shrink-0 px-3 py-1.5 border-t border-surface-3 flex items-center justify-between text-xs text-slate-400">
          <span>{{ pageInfo }}</span>
          <div class="flex gap-1">
            <button @click="prevPage" :disabled="page <= 0"
              class="px-2 py-1 rounded border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 transition">上一页</button>
            <button @click="nextPage" :disabled="(page + 1) * pageSize >= filtered.length"
              class="px-2 py-1 rounded border border-surface-3 hover:bg-surface-2 cursor-pointer disabled:opacity-30 transition">下一页</button>
          </div>
        </div>
      </div>

      <!-- K 线查看器 -->
      <div v-show="showKline" class="flex-1 flex flex-col min-w-0 bg-white">
        <!-- 股票信息头 -->
        <div class="flex-shrink-0 px-4 py-2.5 border-b border-surface-3 flex items-center gap-4">
          <div class="flex items-baseline gap-2 min-w-0">
            <span class="text-base font-bold text-slate-800">{{ selectedSymbol }}</span>
            <span class="text-sm text-slate-400 truncate">{{ selectedName }}</span>
          </div>
          <div class="flex items-baseline gap-2.5">
            <span class="text-lg font-bold text-slate-800 font-mono">{{ quote.close ? quote.close.toFixed(2) : '--' }}</span>
            <span :class="['text-sm font-medium font-mono', chgColor(quote.change)]">
              {{ quote.change > 0 ? '+' : '' }}{{ quote.change ? quote.change.toFixed(2) : '--' }}
            </span>
            <span :class="['text-sm font-medium font-mono', chgColor(quote.change)]">
              {{ quote.change > 0 ? '+' : '' }}{{ quote.changePct || '0' }}%
            </span>
          </div>
          <div class="flex-1"></div>
          <!-- 周期按钮 -->
          <div class="flex items-center gap-0.5">
            <button @click="setPeriod('1min')"
              :class="['period-btn px-2.5 py-1 text-xs font-medium rounded cursor-pointer',
                       period === '1min' ? 'active text-white' : 'text-slate-500 hover:bg-surface-2']">
              1分
            </button>
            <span class="mx-0.5 w-px h-3.5 bg-surface-3"></span>
            <button v-for="p in [{v:'D',l:'日K'},{v:'W',l:'周K'},{v:'M',l:'月K'}]" :key="p.v"
                    @click="setPeriod(p.v)"
              :class="['period-btn px-2.5 py-1 text-xs font-medium rounded cursor-pointer',
                       period === p.v ? 'active text-white' : 'text-slate-500 hover:bg-surface-2']">
              {{ p.l }}
            </button>
          </div>
          <button @click="closeKline" class="p-1 rounded hover:bg-surface-2 cursor-pointer transition ml-1">
            <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
        <!-- 均线图例 -->
        <div class="flex-shrink-0 px-4 py-1 flex items-center gap-4 text-xs border-b border-surface-3">
          <span class="flex items-center gap-1">
            <span class="inline-block w-3 h-0.5 rounded" style="background:#F59E0B"></span>
            <span class="text-slate-500">MA5</span>
          </span>
          <span class="flex items-center gap-1">
            <span class="inline-block w-3 h-0.5 rounded" style="background:#3B82F6"></span>
            <span class="text-slate-500">MA10</span>
          </span>
          <span class="flex items-center gap-1">
            <span class="inline-block w-3 h-0.5 rounded" style="background:#A855F7"></span>
            <span class="text-slate-500">MA20</span>
          </span>
          <span class="flex-1"></span>
          <span v-if="loadingMin" class="text-brand-500 flex items-center gap-1">
            <svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            加载中
          </span>
          <span v-if="period === '1min' && !loadingMin" class="text-xs text-slate-400">
            滚轮平移 · Ctrl+滚轮缩放
          </span>
        </div>
        <!-- K 线图表 -->
        <div id="kline-chart" class="flex-1 min-h-0"></div>
      </div>
    </div>
  </div>
</template>
