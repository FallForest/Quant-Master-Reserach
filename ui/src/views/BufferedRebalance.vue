<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../utils/api'
import { fmtAmount, fmtPct, sideText, sideClass } from '../utils/format'
import { useExecutionDraft } from '../composables/useExecutionDraft'

const router = useRouter()

const props = defineProps({
  embedded: {
    type: Boolean,
    default: false,
  },
})

const loading = ref(true)
const error = ref('')
const payload = ref(null)
const form = ref({
  date: '',
  topK: 55,
  holdTopk: 85,
  riskDegree: 0.95,
  weightMode: 'equal',
  strategyType: 'buffered_weight',
  rebalanceMode: 'weekly',
  nDrop: 4,
})

async function loadPreview() {
  loading.value = true
  error.value = ''
  const p = form.value
  const params = new URLSearchParams({
    strategy_type: p.strategyType,
    rebalance_mode: p.rebalanceMode,
    risk_degree: String(p.riskDegree),
  })
  if (p.date) params.set('date', p.date)
  if (p.strategyType === 'buffered_weight') {
    params.set('top_k', String(p.topK))
    params.set('hold_topk', String(p.holdTopk))
    params.set('weight_mode', p.weightMode)
  } else {
    params.set('top_k', String(p.topK))
    params.set('n_drop', String(p.nDrop))
  }
  const data = await api(`/api/strategy-buffered-rebalance?${params.toString()}`)
  if (!data || data.error) {
    error.value = data?.error || '加载 Buffered 调仓预览失败'
    loading.value = false
    return
  }
  payload.value = data
  if (!form.value.date && data.tradeDate) form.value.date = data.tradeDate
  loading.value = false
}

const { saveDraft, navigateToExecution } = useExecutionDraft({ router })
const sendingToExecution = ref(false)

async function sendToExecution() {
  if (!payload.value?.trades?.length) return
  sendingToExecution.value = true
  try {
    saveDraft(
      'buffered-rebalance',
      payload.value.alias,
      payload.value.tradeDate,
      payload.value.config,
      payload.value.trades,
      payload.value.summary,
    )
    await navigateToExecution()
  } finally {
    sendingToExecution.value = false
  }
}

onMounted(loadPreview)

const summaryCards = computed(() => {
  const summary = payload.value?.summary || {}
  return [
    { label: '保留旧仓', value: summary.keptCount ?? '--', desc: '仍在 buffer 内继续持有', icon: 'M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z', color: 'emerald' },
    { label: '新增标的', value: summary.newCount ?? '--', desc: '本次新进入目标池', icon: 'M12 4.5v15m7.5-7.5h-15', color: 'blue' },
    { label: '预计买入', value: fmtAmount(summary.estimatedBuyAmount), desc: '按目标权重估算', icon: 'M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941', color: 'bull' },
    { label: '预计卖出', value: fmtAmount(summary.estimatedSellAmount), desc: '按目标权重估算', icon: 'M2.25 6L9 12.75l4.286-4.286a11.948 11.948 0 014.306 6.43l.776 2.898m0 0l3.182-5.511m-3.182 5.51l-5.511-3.181', color: 'bear' },
    { label: '预计费用', value: fmtAmount(summary.estimatedFees), desc: '佣金 + 印花税 + 过户费', icon: 'M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 00-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 01-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 003 15h-.75M15 10.5a3 3 0 11-6 0 3 3 0 016 0zm3 0h.008v.008H18V10.5zm-12 0h.008v.008H6V10.5z', color: 'amber' },
    { label: '预计换手', value: summary.turnoverPct != null ? `${summary.turnoverPct}%` : '--', desc: '买卖总额 / 总资产', icon: 'M19.5 12c0-1.232-.046-2.453-.138-3.662a4.006 4.006 0 00-3.7-3.7 48.678 48.678 0 00-7.324 0 4.006 4.006 0 00-3.7 3.7c-.017.22-.032.441-.046.662M19.5 12l3-3m-3 3l-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.006 4.006 0 003.7 3.7 48.656 48.656 0 007.324 0 4.006 4.006 0 003.7-3.7c.017-.22.032-.441.046-.662M4.5 12l3 3m-3-3l-3 3', color: 'purple' },
  ]
})

/**
 * Map each reason string to a semantic color + icon for visual scanning.
 * Matches by keyword since reasons follow a fixed pattern from the backend.
 */
const reasonMeta = (reason) => {
  if (!reason) return null
  if (reason.includes('新入选')) return {
    icon: 'M12 4.5v15m7.5-7.5h-15',
    cls: 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-l-blue-400 dark:border-l-blue-500',
  }
  if (reason.includes('增持')) return {
    icon: 'M12 4.5v15m0 0l6.75-6.75M12 19.5l-6.75-6.75',
    cls: 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 border-l-emerald-400 dark:border-l-emerald-500',
  }
  if (reason.includes('清仓')) return {
    icon: 'M15 12H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z',
    cls: 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border-l-red-400 dark:border-l-red-500',
  }
  if (reason.includes('减持')) return {
    icon: 'M19.5 12h-15',
    cls: 'bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 border-l-orange-400 dark:border-l-orange-500',
  }
  if (reason.includes('无需')) return {
    icon: 'M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
    cls: 'bg-slate-50 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400 border-l-slate-300 dark:border-l-slate-600',
  }
  // fallback
  return {
    icon: 'M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0l3-3m-3 3l-3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z',
    cls: 'bg-gray-50 dark:bg-gray-800/30 text-gray-600 dark:text-gray-400 border-l-gray-300 dark:border-l-gray-600',
  }
}

const cardColorMap = {
  emerald: { bg: 'bg-emerald-50 dark:bg-emerald-900/20', icon: 'text-emerald-500', border: 'border-emerald-200/60 dark:border-emerald-800/30' },
  blue: { bg: 'bg-blue-50 dark:bg-blue-900/20', icon: 'text-blue-500', border: 'border-blue-200/60 dark:border-blue-800/30' },
  bull: { bg: 'bg-red-50 dark:bg-red-900/20', icon: 'text-bull', border: 'border-red-200/60 dark:border-red-800/30' },
  bear: { bg: 'bg-emerald-50 dark:bg-emerald-900/20', icon: 'text-bear', border: 'border-emerald-200/60 dark:border-emerald-800/30' },
  amber: { bg: 'bg-amber-50 dark:bg-amber-900/20', icon: 'text-amber-500', border: 'border-amber-200/60 dark:border-amber-800/30' },
  purple: { bg: 'bg-purple-50 dark:bg-purple-900/20', icon: 'text-purple-500', border: 'border-purple-200/60 dark:border-purple-800/30' },
}

function onStrategyChange() {
  if (form.value.strategyType === 'buffered_weight') {
    form.value.topK = 55
    form.value.holdTopk = 85
    form.value.weightMode = 'equal'
    form.value.rebalanceMode = 'weekly'
  } else {
    form.value.topK = 45
    form.value.nDrop = 4
  }
  loadPreview()
}

function runPreview() {
  loadPreview()
}

// Auto-load when params change with debounce
watch(() => [form.value.strategyType, form.value.rebalanceMode], () => {
  // strategy change handled by onStrategyChange, rebalance just reloads
  if (payload.value) loadPreview()
})
</script>

<template>
  <div :class="[props.embedded ? 'space-y-5' : 'p-4 sm:p-6 space-y-5 animate-slide-in']">

    <!-- Header Bar -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-br from-brand-500 to-blue-600 flex items-center justify-center shadow-md shadow-brand-500/15">
          <svg class="w-4.5 h-4.5 text-white" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h10.5m-10.5 5.25h16.5" />
          </svg>
        </div>
        <div>
          <h2 class="text-base font-semibold text-slate-800 dark:text-slate-100">{{ payload?.strategyRef?.strategyLabel || '策略调仓' }}</h2>
          <p class="text-xs text-slate-500 mt-0.5">{{ payload?.strategyRef?.strategyDesc || '选择策略并生成调仓建议' }}</p>
        </div>
      </div>
      <div class="flex-1"></div>
      <div class="flex flex-wrap gap-2">
        <button
          class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white dark:bg-slate-700 border border-surface-3 dark:border-slate-600 text-slate-700 dark:text-slate-200 text-sm font-medium hover:bg-surface-2 dark:hover:bg-slate-600 transition-all duration-200 cursor-pointer shadow-sm"
          @click="runPreview"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
          </svg>
          重新生成
        </button>
        <button
          data-testid="buffered-to-execution"
          class="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-gradient-to-r from-cta to-amber-500 text-white text-sm font-semibold hover:from-cta/90 hover:to-amber-500/90 transition-all duration-200 cursor-pointer shadow-md shadow-cta/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
          :disabled="sendingToExecution || !payload?.trades?.length"
          @click="sendToExecution"
        >
          <svg v-if="!sendingToExecution" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
          </svg>
          <svg v-else class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {{ sendingToExecution ? '跳转中...' : '前往执行' }}
        </button>
      </div>
    </div>

    <!-- Parameters Form -->
    <div class="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 shadow-sm overflow-hidden">
      <div class="px-5 py-3.5 border-b border-surface-3/60 dark:border-slate-700/40 bg-surface-2/30 dark:bg-slate-800/30">
        <div class="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
          <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
          </svg>
          参数配置
        </div>
      </div>
      <div class="p-5">
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-4">
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1.5">策略类型</label>
            <select v-model="form.strategyType" @change="onStrategyChange" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none cursor-pointer transition-all duration-200">
              <option value="buffered_weight">BufferedWeight (SOTA)</option>
              <option value="topk_dropout">TopkDropout (基准)</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1.5">目标日期</label>
            <input v-model="form.date" type="date" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none transition-all duration-200" />
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1.5">topK</label>
            <input v-model.number="form.topK" type="number" min="1" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none font-mono transition-all duration-200" />
          </div>
          <template v-if="form.strategyType === 'buffered_weight'">
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1.5">hold_topk</label>
              <input v-model.number="form.holdTopk" type="number" min="1" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none font-mono transition-all duration-200" />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1.5">再平衡</label>
              <select v-model="form.rebalanceMode" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none cursor-pointer transition-all duration-200">
                <option value="weekly">每周（SOTA）</option>
                <option value="daily">每日</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1.5">权重模式</label>
              <select v-model="form.weightMode" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none cursor-pointer transition-all duration-200">
                <option value="equal">equal（SOTA）</option>
                <option value="score">score</option>
              </select>
            </div>
          </template>
          <template v-else>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1.5">n_drop</label>
              <input v-model.number="form.nDrop" type="number" min="1" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none font-mono transition-all duration-200" />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1.5">风险敞口</label>
              <input v-model.number="form.riskDegree" type="number" min="0" max="1" step="0.01" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none font-mono transition-all duration-200" />
            </div>
          </template>
          <div v-if="form.strategyType === 'buffered_weight'">
            <label class="block text-xs font-medium text-slate-500 mb-1.5">风险敞口</label>
            <input v-model.number="form.riskDegree" type="number" min="0" max="1" step="0.01" class="w-full px-3 py-2.5 text-sm rounded-xl border border-surface-3 dark:border-slate-600 bg-white dark:bg-slate-700 dark:text-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none font-mono transition-all duration-200" />
          </div>
        </div>
        <div class="mt-3 flex items-start gap-2 text-[11px] text-slate-500 bg-surface-2/40 dark:bg-slate-700/30 rounded-lg px-3 py-2" v-if="form.strategyType === 'buffered_weight'">
          <svg class="w-3.5 h-3.5 text-brand-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
          </svg>
          <span>hold_topk - topk = buffer 大小（SOTA: 85-55=30）；旧持仓若仍在该缓冲排名内，优先保留，减少不必要换手。</span>
        </div>
        <div class="mt-3 flex items-start gap-2 text-[11px] text-slate-500 bg-surface-2/40 dark:bg-slate-700/30 rounded-lg px-3 py-2" v-else>
          <svg class="w-3.5 h-3.5 text-brand-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
          </svg>
          <span>每日按模型分数排名取前 topk 只，其余全部卖出。n_drop 控制每日最多轮出数量（SOTA: 45/4）。</span>
        </div>
      </div>
    </div>

    <!-- Error Alert -->
    <div v-if="error" class="rounded-xl border border-danger/20 bg-danger/5 backdrop-blur-sm px-4 py-3 text-sm text-danger flex items-start gap-2">
      <svg class="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
      </svg>
      {{ error }}
    </div>

    <!-- Strategy Reference Banner -->
    <div v-if="payload?.strategyRef" class="rounded-xl border border-amber-200/60 dark:border-amber-800/30 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/10 backdrop-blur-sm px-5 py-3.5 text-xs text-amber-800 dark:text-amber-300 flex flex-wrap items-center gap-2">
      <svg class="w-4 h-4 text-amber-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
      </svg>
      <span class="font-semibold">{{ payload.strategyRef.strategyLabel }}</span>
      <span class="text-amber-400 dark:text-amber-600">·</span>
      <span>项目验证：</span>
      <span class="font-semibold text-amber-900 dark:text-amber-200">IR {{ payload.strategyRef.provenIR }}</span>
      <span class="text-amber-400 dark:text-amber-600">·</span>
      <span class="font-semibold text-amber-900 dark:text-amber-200">年化 {{ payload.strategyRef.provenAnnRet }}</span>
      <span class="text-amber-400 dark:text-amber-600">·</span>
      <span>{{ payload.strategyRef.source }}</span>
      <span v-if="payload.rebalanceMode === 'weekly'" class="ml-1 inline-flex px-2 py-0.5 rounded-full bg-amber-200/60 dark:bg-amber-800/30 text-amber-800 dark:text-amber-300 font-medium">每周再平衡</span>
      <span v-else class="ml-1 inline-flex px-2 py-0.5 rounded-full bg-amber-200/60 dark:bg-amber-800/30 text-amber-800 dark:text-amber-300 font-medium">每日再平衡</span>
      <span v-if="payload.nextRebalanceDate" class="ml-auto text-amber-600 dark:text-amber-400 font-medium">下次调仓：{{ payload.nextRebalanceDate }}</span>
    </div>

    <!-- Loading Skeleton -->
    <template v-if="loading">
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div v-for="i in 6" :key="i" class="bg-white/80 dark:bg-slate-800/50 rounded-2xl border border-surface-3/80 dark:border-slate-700/50 p-4">
          <div class="flex items-center gap-2 mb-3">
            <div class="skeleton w-8 h-8 rounded-lg"></div>
            <div class="skeleton h-3 w-16"></div>
          </div>
          <div class="skeleton h-7 w-20 mb-1.5"></div>
          <div class="skeleton h-2.5 w-24"></div>
        </div>
      </div>
    </template>

    <!-- Main Content -->
    <template v-else-if="payload">

      <!-- Explanation Banner -->
      <div class="rounded-xl border border-brand-200/60 dark:border-brand-800/30 bg-gradient-to-r from-brand-50 to-blue-50/50 dark:from-brand-950/30 dark:to-blue-950/10 backdrop-blur-sm px-5 py-4">
        <div class="flex items-start gap-3">
          <div class="w-8 h-8 rounded-lg bg-brand-100 dark:bg-brand-900/40 flex items-center justify-center flex-shrink-0">
            <svg class="w-4 h-4 text-brand-600 dark:text-brand-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
            </svg>
          </div>
          <div>
            <div class="text-sm font-semibold text-brand-800 dark:text-brand-200">{{ payload.explanation?.title }}</div>
            <div class="text-sm text-brand-700 dark:text-brand-300 mt-1 leading-relaxed">{{ payload.explanation?.why }}</div>
          </div>
        </div>
      </div>

      <!-- Summary Cards -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div
          v-for="card in summaryCards"
          :key="card.label"
          :class="[
            'group rounded-2xl border backdrop-blur-sm p-4 transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 cursor-default',
            'bg-white/80 dark:bg-slate-800/50',
            cardColorMap[card.color]?.border || 'border-surface-3/80 dark:border-slate-700/50',
          ]"
        >
          <div class="flex items-center gap-2 mb-2.5">
            <div :class="['w-8 h-8 rounded-lg flex items-center justify-center transition-colors duration-200', cardColorMap[card.color]?.bg || 'bg-surface-2']">
              <svg :class="['w-4 h-4', cardColorMap[card.color]?.icon || 'text-slate-500']" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" :d="card.icon" />
              </svg>
            </div>
            <div class="text-[11px] font-medium text-slate-500">{{ card.label }}</div>
          </div>
          <div class="text-xl font-bold font-mono text-slate-800 dark:text-slate-100 break-all leading-tight">{{ card.value }}</div>
          <div class="text-[10px] text-slate-400 mt-1">{{ card.desc }}</div>
        </div>
      </div>

      <!-- Trades + Strategy Explanation -->
      <div class="grid grid-cols-1 xl:grid-cols-5 gap-4">

        <!-- Trades Table -->
        <div class="xl:col-span-3 bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 shadow-sm overflow-hidden">
          <div class="px-5 py-3.5 border-b border-surface-3/60 dark:border-slate-700/40 bg-surface-2/30 dark:bg-slate-800/30 flex items-center justify-between">
            <div class="flex items-center gap-2">
              <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
              </svg>
              <div>
                <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-200">调仓建议</h3>
                <p class="text-[10px] text-slate-500 mt-0.5">按当前持仓与目标权重对比，输出预计买卖动作</p>
              </div>
            </div>
            <div class="text-xs text-slate-500 font-mono bg-surface-2/60 dark:bg-slate-700/50 px-2.5 py-1 rounded-lg">{{ payload.tradeDate }}</div>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider bg-surface-2/40 dark:bg-slate-700/20">
                  <th class="py-2.5 px-4">代码</th>
                  <th class="py-2.5 px-3">名称</th>
                  <th class="py-2.5 px-3">动作</th>
                  <th class="py-2.5 px-3">换仓原因</th>
                  <th class="py-2.5 px-3 text-right">现持仓</th>
                  <th class="py-2.5 px-3 text-right">目标持仓</th>
                  <th class="py-2.5 px-3 text-right">权重变化</th>
                  <th class="py-2.5 px-4 text-right">预计金额</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.trades" :key="row.instrument" class="border-b border-surface-3/40 dark:border-slate-700/30 last:border-0 hover:bg-brand-50/40 dark:hover:bg-brand-900/10 transition-colors duration-150">
                  <td class="py-2.5 px-4 font-mono text-xs font-medium text-slate-700 dark:text-slate-300">{{ row.instrument }}</td>
                  <td class="py-2.5 px-3 text-xs text-slate-600 dark:text-slate-400">{{ row.name }}</td>
                  <td class="py-2.5 px-3">
                    <span :class="['inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-semibold', sideClass(row.side)]">
                      <svg v-if="row.side === 'buy'" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
                      </svg>
                      <svg v-else-if="row.side === 'sell'" class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 4.5l15 15m0 0V8.25m0 11.25H8.25" />
                      </svg>
                      {{ sideText(row.side) }}
                    </span>
                  </td>
                  <td class="py-2.5 px-3">
                    <template v-if="row.reason">
                      <span
                        class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium border-l-2 whitespace-nowrap max-w-full"
                        :class="reasonMeta(row.reason)?.cls || 'bg-slate-50 dark:bg-slate-800/30 text-slate-600 dark:text-slate-400'"
                      >
                        <svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" :d="reasonMeta(row.reason)?.icon || ''" />
                        </svg>
                        <span class="truncate">{{ row.reason }}</span>
                        <span
                          v-if="row.side === 'buy'"
                          class="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-bull/60"
                        ></span>
                        <span
                          v-else-if="row.side === 'sell'"
                          class="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-bear/60"
                        ></span>
                      </span>
                    </template>
                    <span v-else class="text-xs text-slate-400">--</span>
                  </td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs text-slate-700 dark:text-slate-300">{{ row.currentShares.toLocaleString() }}</td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs text-slate-700 dark:text-slate-300">{{ row.targetShares.toLocaleString() }}</td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs font-medium" :class="row.targetWeight >= row.currentWeight ? 'text-bull' : 'text-bear'">
                    {{ fmtPct(row.targetWeight - row.currentWeight) }}
                  </td>
                  <td class="py-2.5 px-4 text-right font-mono text-xs text-slate-700 dark:text-slate-300">{{ fmtAmount(row.tradeAmount) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Strategy Explanation Panel -->
        <div class="xl:col-span-2 bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 shadow-sm overflow-hidden">
          <div class="px-5 py-3.5 border-b border-surface-3/60 dark:border-slate-700/40 bg-surface-2/30 dark:bg-slate-800/30 flex items-center gap-2">
            <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
            <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-200">策略解释</h3>
          </div>
          <div class="p-5 space-y-4 text-sm text-slate-600 dark:text-slate-400">
            <!-- Model Source -->
            <div class="rounded-xl bg-surface-2/50 dark:bg-slate-700/30 border border-surface-3/60 dark:border-slate-700/40 px-4 py-3.5">
              <div class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">模型来源</div>
              <div class="font-semibold text-slate-700 dark:text-slate-200 break-all">{{ payload.alias || 'sample' }}</div>
              <div class="flex flex-wrap gap-2 mt-2">
                <span v-if="payload.config.topK != null" class="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-100/60 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-[11px] font-mono font-medium">topk={{ payload.config.topK }}</span>
                <span v-if="payload.config.holdTopk != null" class="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-100/60 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-[11px] font-mono font-medium">hold_topk={{ payload.config.holdTopk }}</span>
                <span v-if="payload.config.rankBuffer != null" class="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-100/60 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-[11px] font-mono font-medium">buffer={{ payload.config.rankBuffer }}</span>
                <span v-if="payload.config.nDrop != null" class="inline-flex items-center px-2 py-0.5 rounded-md bg-brand-100/60 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-[11px] font-mono font-medium">n_drop={{ payload.config.nDrop }}</span>
              </div>
            </div>

            <!-- Execution Logic -->
            <div class="rounded-xl border border-surface-3/60 dark:border-slate-700/40 px-4 py-3.5 bg-white/60 dark:bg-slate-800/30">
              <div class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2.5">执行逻辑</div>
              <ul class="text-xs text-slate-600 dark:text-slate-400 space-y-2">
                <li v-for="item in payload.explanation?.how || []" :key="item" class="flex items-start gap-2">
                  <span class="w-1.5 h-1.5 rounded-full bg-brand-400 mt-1.5 flex-shrink-0"></span>
                  {{ item }}
                </li>
              </ul>
            </div>

            <!-- Note: auto-cap notification -->
            <div
              v-if="payload.explanation?.note"
              class="rounded-xl border border-amber-200/60 dark:border-amber-800/30 bg-gradient-to-r from-amber-50 to-orange-50/50 dark:from-amber-900/20 dark:to-orange-900/10 px-4 py-3 text-xs text-amber-800 dark:text-amber-300 flex items-start gap-2"
            >
              <svg class="w-4 h-4 flex-shrink-0 mt-0.5 text-amber-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
              </svg>
              <span>{{ payload.explanation.note }}</span>
            </div>

            <!-- Cash & Fees -->
            <div class="rounded-xl border border-surface-3/60 dark:border-slate-700/40 px-4 py-3.5 bg-white/60 dark:bg-slate-800/30">
              <div class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2.5">现金与费用</div>
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div class="flex flex-col gap-0.5">
                  <span class="text-slate-500">当前现金</span>
                  <span class="font-mono font-semibold text-slate-700 dark:text-slate-200">{{ fmtAmount(payload.holdings.cash) }}</span>
                </div>
                <div class="flex flex-col gap-0.5">
                  <span class="text-slate-500">调仓后现金</span>
                  <span class="font-mono font-semibold text-slate-700 dark:text-slate-200">{{ fmtAmount(payload.summary.cashAfterTrades) }}</span>
                </div>
                <div class="flex flex-col gap-0.5">
                  <span class="text-slate-500">预计费用</span>
                  <span class="font-mono font-semibold text-slate-700 dark:text-slate-200">{{ fmtAmount(payload.summary.estimatedFees) }}</span>
                </div>
                <div class="flex flex-col gap-0.5">
                  <span class="text-slate-500">风险敞口</span>
                  <span class="font-mono font-semibold text-slate-700 dark:text-slate-200">{{ Math.round(payload.config.riskDegree * 100) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom Tables: Target Positions + Model Hit -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">

        <!-- Target Positions Comparison -->
        <div class="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 shadow-sm overflow-hidden">
          <div class="px-5 py-3.5 border-b border-surface-3/60 dark:border-slate-700/40 bg-surface-2/30 dark:bg-slate-800/30 flex items-center gap-2">
            <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
            </svg>
            <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-200">目标持仓前后对比</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider bg-surface-2/40 dark:bg-slate-700/20">
                  <th class="py-2.5 px-4">代码</th>
                  <th class="py-2.5 px-3">buffer 保留</th>
                  <th class="py-2.5 px-3 text-right">现权重</th>
                  <th class="py-2.5 px-3 text-right">目标权重</th>
                  <th class="py-2.5 px-4 text-right">分数</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.targetPositions" :key="row.instrument" class="border-b border-surface-3/40 dark:border-slate-700/30 last:border-0 hover:bg-brand-50/40 dark:hover:bg-brand-900/10 transition-colors duration-150">
                  <td class="py-2.5 px-4 font-mono text-xs font-medium text-slate-700 dark:text-slate-300">{{ row.instrument }}</td>
                  <td class="py-2.5 px-3">
                    <span v-if="row.bufferKept" class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[11px] font-semibold">
                      <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                      </svg>
                      保留
                    </span>
                    <span v-else-if="row.isNew" class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-[11px] font-semibold">
                      <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                      </svg>
                      新增
                    </span>
                    <span v-else class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-2 dark:bg-slate-700 text-slate-500 text-[11px] font-medium">否</span>
                  </td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs text-slate-700 dark:text-slate-300">{{ row.currentWeight.toFixed(2) }}%</td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs text-slate-700 dark:text-slate-300">{{ row.targetWeight.toFixed(2) }}%</td>
                  <td class="py-2.5 px-4 text-right font-mono text-xs font-semibold text-brand-700 dark:text-brand-400">{{ row.score ?? '--' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Model Stock Selection Hit -->
        <div class="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 shadow-sm overflow-hidden">
          <div class="px-5 py-3.5 border-b border-surface-3/60 dark:border-slate-700/40 bg-surface-2/30 dark:bg-slate-800/30 flex items-center gap-2">
            <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
            <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-200">模型选股命中</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider bg-surface-2/40 dark:bg-slate-700/20">
                  <th class="py-2.5 px-4">排名</th>
                  <th class="py-2.5 px-3">代码 / 名称</th>
                  <th class="py-2.5 px-3 text-right">score</th>
                  <th class="py-2.5 px-4">结果</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.prediction.stocks" :key="row.instrument" class="border-b border-surface-3/40 dark:border-slate-700/30 last:border-0 hover:bg-brand-50/40 dark:hover:bg-brand-900/10 transition-colors duration-150">
                  <td class="py-2.5 px-4">
                    <span class="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-surface-2/60 dark:bg-slate-700/50 font-mono text-xs font-bold text-slate-600 dark:text-slate-300">
                      {{ row.rank }}
                    </span>
                  </td>
                  <td class="py-2.5 px-3">
                    <div class="font-mono text-xs font-medium text-slate-700 dark:text-slate-300">{{ row.instrument }}</div>
                    <div class="text-[11px] text-slate-500 mt-0.5">{{ row.name }}</div>
                  </td>
                  <td class="py-2.5 px-3 text-right font-mono text-xs font-semibold text-brand-700 dark:text-brand-400">{{ row.score }}</td>
                  <td class="py-2.5 px-4">
                    <span v-if="payload.selected.some(item => item.instrument === row.instrument)" class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[11px] font-semibold">
                      <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      进入目标仓
                    </span>
                    <span v-else class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-2/80 dark:bg-slate-700/50 text-slate-500 text-[11px] font-medium">
                      未入选
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
