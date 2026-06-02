<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../utils/api'

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
const sendingToExecution = ref(false)

const form = ref({
  date: '',
  topK: 5,
  holdTopk: 8,
  riskDegree: 0.95,
  weightMode: 'equal',
})

async function loadPreview() {
  loading.value = true
  error.value = ''
  const params = new URLSearchParams({
    top_k: String(form.value.topK),
    hold_topk: String(form.value.holdTopk),
    risk_degree: String(form.value.riskDegree),
    weight_mode: form.value.weightMode,
  })
  if (form.value.date) params.set('date', form.value.date)
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

async function sendToExecution() {
  if (!payload.value?.trades?.length) return
  sendingToExecution.value = true
  try {
    sessionStorage.setItem('executionDraft', JSON.stringify({
      source: 'buffered-rebalance',
      alias: payload.value.alias,
      tradeDate: payload.value.tradeDate,
      config: payload.value.config,
      trades: payload.value.trades,
      summary: payload.value.summary,
    }))
    await router.push('/execution')
  } finally {
    sendingToExecution.value = false
  }
}

onMounted(loadPreview)

const summaryCards = computed(() => {
  const summary = payload.value?.summary || {}
  return [
    { label: '保留旧仓', value: summary.keptCount ?? '--', desc: '仍在 buffer 内继续持有' },
    { label: '新增标的', value: summary.newCount ?? '--', desc: '本次新进入目标池' },
    { label: '预计买入', value: fmtAmount(summary.estimatedBuyAmount), desc: '按目标权重估算' },
    { label: '预计卖出', value: fmtAmount(summary.estimatedSellAmount), desc: '按目标权重估算' },
    { label: '预计费用', value: fmtAmount(summary.estimatedFees), desc: '佣金 + 印花税 + 过户费' },
    { label: '预计换手', value: summary.turnoverPct != null ? `${summary.turnoverPct}%` : '--', desc: '买卖总额 / 总资产' },
  ]
})

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
  return `${sign}${Number(n).toFixed(2)}%`
}

function sideText(side) {
  return side === 'buy' ? '买入' : side === 'sell' ? '卖出' : '持有'
}

function sideClass(side) {
  return side === 'buy'
    ? 'bg-bull/10 text-bull'
    : side === 'sell'
      ? 'bg-bear/10 text-bear'
      : 'bg-surface-2 text-slate-500'
}

function runPreview() {
  loadPreview()
}
</script>

<template>
  <div :class="[props.embedded ? 'space-y-5' : 'p-4 sm:p-6 space-y-5 animate-slide-in']">
    <div class="flex flex-wrap items-start gap-3">
      <div>
        <h2 class="text-base font-semibold text-slate-700">Buffered 调仓</h2>
        <p class="text-xs text-slate-500 mt-1">基于 buffer 保留机制，在降低不必要换手的同时完成目标仓位调整。</p>
      </div>
      <div class="flex-1"></div>
      <div class="flex flex-wrap gap-2">
        <button
          class="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 transition cursor-pointer"
          @click="runPreview"
        >
          重新生成调仓建议
        </button>
        <button
          data-testid="buffered-to-execution"
          class="px-4 py-2 rounded-lg bg-cta text-white text-sm font-medium hover:bg-cta/90 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="sendingToExecution || !payload?.trades?.length"
          @click="sendToExecution"
        >
          {{ sendingToExecution ? '跳转中...' : '前往执行' }}
        </button>
      </div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-4">
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">目标日期</label>
          <input v-model="form.date" type="date" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none" />
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">topk</label>
          <input v-model.number="form.topK" type="number" min="1" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" />
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">hold_topk</label>
          <input v-model.number="form.holdTopk" type="number" min="1" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" />
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">risk degree</label>
          <input v-model.number="form.riskDegree" type="number" min="0" max="1" step="0.01" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" />
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">权重模式</label>
          <select v-model="form.weightMode" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer">
            <option value="equal">equal</option>
            <option value="score">score</option>
          </select>
        </div>
      </div>
      <div class="mt-3 text-[11px] text-slate-500">
        hold_topk - topk = buffer 大小；旧持仓若仍在该缓冲排名内，会优先保留，减少不必要换手。
      </div>
    </div>

    <div v-if="error" class="rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-sm text-danger">
      {{ error }}
    </div>

    <template v-if="loading">
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div v-for="i in 6" :key="i" class="bg-white rounded-xl border border-surface-3 p-3.5">
          <div class="skeleton h-3 w-16 mb-2"></div>
          <div class="skeleton h-6 w-24 mb-1"></div>
          <div class="skeleton h-2.5 w-20"></div>
        </div>
      </div>
    </template>

    <template v-else-if="payload">
      <div class="rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-700">
        <div class="font-medium">{{ payload.explanation?.title }}</div>
        <div class="mt-1">{{ payload.explanation?.why }}</div>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div v-for="card in summaryCards" :key="card.label" class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
          <div class="text-[11px] text-slate-500 mb-1">{{ card.label }}</div>
          <div class="text-xl font-bold font-mono text-slate-800 break-all">{{ card.value }}</div>
          <div class="text-[10px] text-slate-500 mt-0.5">{{ card.desc }}</div>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div class="xl:col-span-3 bg-white rounded-xl border border-surface-3 p-4">
          <div class="flex items-center justify-between mb-3">
            <div>
              <h3 class="text-sm font-semibold text-slate-600">调仓建议</h3>
              <p class="text-[10px] text-slate-500 mt-1">按当前持仓与目标权重对比，输出预计买卖动作。</p>
            </div>
            <div class="text-xs text-slate-500">trade date: {{ payload.tradeDate }}</div>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">代码</th>
                  <th class="py-2 pr-3">名称</th>
                  <th class="py-2 pr-3">动作</th>
                  <th class="py-2 pr-3 text-right">现持仓</th>
                  <th class="py-2 pr-3 text-right">目标持仓</th>
                  <th class="py-2 pr-3 text-right">权重变化</th>
                  <th class="py-2 pr-3 text-right">预计金额</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.trades" :key="row.instrument" class="border-b border-surface-3/50 last:border-0 hover:bg-surface-1/70 transition">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.instrument }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.name }}</td>
                  <td class="py-2.5 pr-3">
                    <span :class="['inline-flex px-2 py-1 rounded-full text-[11px] font-medium', sideClass(row.side)]">{{ sideText(row.side) }}</span>
                  </td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.currentShares.toLocaleString() }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.targetShares.toLocaleString() }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs" :class="row.targetWeight >= row.currentWeight ? 'text-bull' : 'text-bear'">
                    {{ fmtPct(row.targetWeight - row.currentWeight) }}
                  </td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ fmtAmount(row.tradeAmount) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="xl:col-span-2 bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">策略解释</h3>
          <div class="space-y-3 text-sm text-slate-600">
            <div class="rounded-lg bg-surface-1/70 border border-surface-3 px-4 py-3">
              <div class="text-xs text-slate-500 mb-1">模型来源</div>
              <div class="font-medium break-all">{{ payload.alias || 'sample' }}</div>
              <div class="text-xs text-slate-500 mt-1">topk={{ payload.config.topK }}, hold_topk={{ payload.config.holdTopk }}, buffer={{ payload.config.rankBuffer }}</div>
            </div>
            <div class="rounded-lg border border-surface-3 px-4 py-3 bg-white">
              <div class="text-xs text-slate-500 mb-2">执行逻辑</div>
              <ul class="text-xs text-slate-600 space-y-1.5 list-disc pl-4">
                <li v-for="item in payload.explanation?.how || []" :key="item">{{ item }}</li>
              </ul>
            </div>
            <div class="rounded-lg border border-surface-3 px-4 py-3 bg-white">
              <div class="text-xs text-slate-500 mb-2">现金与费用</div>
              <div class="grid grid-cols-2 gap-y-2 gap-x-4 text-xs">
                <div>当前现金：<span class="font-mono text-slate-700">{{ fmtAmount(payload.holdings.cash) }}</span></div>
                <div>调仓后现金：<span class="font-mono text-slate-700">{{ fmtAmount(payload.summary.cashAfterTrades) }}</span></div>
                <div>预计费用：<span class="font-mono text-slate-700">{{ fmtAmount(payload.summary.estimatedFees) }}</span></div>
                <div>风险敞口：<span class="font-mono text-slate-700">{{ Math.round(payload.config.riskDegree * 100) }}%</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">目标持仓前后对比</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">代码</th>
                  <th class="py-2 pr-3">buffer 保留</th>
                  <th class="py-2 pr-3 text-right">现权重</th>
                  <th class="py-2 pr-3 text-right">目标权重</th>
                  <th class="py-2 pr-3 text-right">分数</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.targetPositions" :key="row.instrument" class="border-b border-surface-3/50 last:border-0">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.instrument }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.bufferKept ? '是' : (row.isNew ? '新增' : '否') }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.currentWeight.toFixed(2) }}%</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.targetWeight.toFixed(2) }}%</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-brand-700">{{ row.score ?? '--' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="bg-white rounded-xl border border-surface-3 p-4">
          <h3 class="text-sm font-semibold text-slate-600 mb-3">模型选股命中</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">排名</th>
                  <th class="py-2 pr-3">代码 / 名称</th>
                  <th class="py-2 pr-3 text-right">score</th>
                  <th class="py-2 pr-3">结果</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in payload.prediction.stocks" :key="row.instrument" class="border-b border-surface-3/50 last:border-0">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-500">#{{ row.rank }}</td>
                  <td class="py-2.5 pr-3">
                    <div class="font-mono text-xs text-slate-700">{{ row.instrument }}</div>
                    <div class="text-[11px] text-slate-500">{{ row.name }}</div>
                  </td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-brand-700">{{ row.score }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">
                    <span v-if="payload.selected.some(item => item.instrument === row.instrument)" class="inline-flex px-2 py-1 rounded-full bg-success/10 text-success">进入目标仓</span>
                    <span v-else class="inline-flex px-2 py-1 rounded-full bg-surface-2 text-slate-500">未入选</span>
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
