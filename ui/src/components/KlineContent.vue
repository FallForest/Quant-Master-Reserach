<script setup>
defineProps({
  symbol: String,
  name: String,
  quote: Object,
  period: String,
  loadingMin: Boolean,
  marketOpen: Boolean,
})
defineEmits(['close', 'set-period'])

function chgColor(value) {
  return value > 0 ? 'text-red-500' : value < 0 ? 'text-emerald-500' : 'text-slate-400'
}

function fmtVol(value) {
  if (!value) return '--'
  if (value >= 1e8) return `${(value / 1e8).toFixed(2)}亿`
  if (value >= 1e4) return `${(value / 1e4).toFixed(1)}万`
  return String(value)
}

function fmtAmt(value) {
  if (!value) return '--'
  if (value >= 1e8) return `${(value / 1e8).toFixed(2)}亿`
  if (value >= 1e4) return `${(value / 1e4).toFixed(1)}万`
  return String(Math.round(value))
}

const periods = [
  { key: '1min', label: '1分' },
  { key: 'D', label: '日' },
  { key: 'W', label: '周' },
  { key: 'M', label: '月' },
]
</script>

<template>
  <div class="flex-1 flex flex-col min-w-0 min-h-0">
    <!-- 顶部工具栏 -->
    <div class="flex-shrink-0 px-5 py-3 border-b border-surface-3 flex flex-wrap items-center gap-4">
      <!-- 股票信息 -->
      <div class="flex items-center gap-2.5 min-w-0">
        <div class="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center flex-shrink-0">
          <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        </div>
        <div class="min-w-0">
          <div class="flex items-baseline gap-2">
            <span class="text-sm font-bold text-slate-800">{{ symbol }}</span>
            <span class="text-xs text-slate-400 truncate">{{ name }}</span>
          </div>
        </div>
      </div>

      <!-- 价格 -->
      <div class="flex items-baseline gap-3">
        <span class="text-xl font-bold text-slate-900 font-mono tabular-nums">
          {{ quote?.close?.toFixed(2) ?? '--' }}
        </span>
        <div class="flex items-baseline gap-2">
          <span :class="['text-sm font-semibold font-mono tabular-nums', chgColor(quote?.change)]">
            {{ (quote?.change ?? 0) > 0 ? '+' : '' }}{{ quote?.change?.toFixed(2) ?? '0.00' }}
          </span>
          <span
            :class="[
              'text-xs font-semibold font-mono px-1.5 py-0.5 rounded',
              (quote?.change ?? 0) > 0
                ? 'bg-red-50 text-red-600'
                : (quote?.change ?? 0) < 0
                  ? 'bg-emerald-50 text-emerald-600'
                  : 'bg-surface-2 text-slate-400',
            ]"
          >
            {{ (quote?.change ?? 0) > 0 ? '+' : '' }}{{ quote?.changePct ?? '0' }}%
          </span>
        </div>
      </div>

      <div class="flex-1" />

      <!-- 周期切换 -->
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
        <template v-for="(item, index) in periods" :key="item.key">
          <span v-if="index === 1" class="w-px h-4 bg-surface-3" />
          <button
            :aria-label="`切换到${item.label}`"
            :class="[
              'px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
              period === item.key ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700',
            ]"
            @click="$emit('set-period', item.key)"
          >
            {{ item.label }}
          </button>
        </template>
      </div>

      <!-- 市场状态 -->
      <span
        :class="[
          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium',
          marketOpen ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400',
        ]"
      >
        <span :class="['w-1.5 h-1.5 rounded-full', marketOpen ? 'bg-emerald-400 animate-pulse' : 'bg-slate-300']" />
        {{ marketOpen ? '交易中' : '休市' }}
      </span>

      <!-- 关闭按钮 -->
      <button
        aria-label="关闭K线"
        class="p-1.5 rounded-lg hover:bg-surface-2 cursor-pointer transition-colors duration-150"
        @click="$emit('close')"
      >
        <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <!-- 行情概要 -->
    <div class="flex-shrink-0 px-5 py-1.5 flex items-center gap-5 text-xs border-b border-surface-3 bg-surface-0/50">
      <span class="text-slate-400">今开</span>
      <span class="font-mono text-slate-700 font-medium">{{ quote?.open?.toFixed(2) ?? '--' }}</span>
      <span class="text-slate-400">最高</span>
      <span :class="['font-mono font-medium', chgColor((quote?.high ?? 0) - (quote?.lastClose ?? 0))]">
        {{ quote?.high?.toFixed(2) ?? '--' }}
      </span>
      <span class="text-slate-400">最低</span>
      <span :class="['font-mono font-medium', chgColor((quote?.low ?? 0) - (quote?.lastClose ?? 0))]">
        {{ quote?.low?.toFixed(2) ?? '--' }}
      </span>
      <span class="text-slate-400">昨收</span>
      <span class="font-mono text-slate-500">{{ quote?.lastClose?.toFixed(2) ?? '--' }}</span>
      <span class="text-slate-400">成交量</span>
      <span class="font-mono text-slate-600">{{ fmtVol(quote?.volume) }}</span>
      <span class="text-slate-400">成交额</span>
      <span class="font-mono text-slate-600">{{ fmtAmt(quote?.amount) }}</span>
    </div>

    <!-- MA 图例 -->
    <div class="flex-shrink-0 px-5 py-1.5 flex items-center gap-5 text-xs border-b border-surface-3">
      <span class="flex items-center gap-1.5">
        <span class="inline-block w-3 h-[2px] rounded-full" style="background:#F59E0B" />
        <span class="text-slate-500 font-medium">MA5</span>
      </span>
      <span class="flex items-center gap-1.5">
        <span class="inline-block w-3 h-[2px] rounded-full" style="background:#3B82F6" />
        <span class="text-slate-500 font-medium">MA10</span>
      </span>
      <span class="flex items-center gap-1.5">
        <span class="inline-block w-3 h-[2px] rounded-full" style="background:#A855F7" />
        <span class="text-slate-500 font-medium">MA20</span>
      </span>
      <div class="flex-1" />
      <span v-if="loadingMin" class="text-brand-500 flex items-center gap-1.5">
        <svg class="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        加载中...
      </span>
    </div>

    <!-- K线图容器 -->
    <div id="kline-chart" class="flex-1 min-h-0" />
  </div>
</template>
