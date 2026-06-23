<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../../utils/api'
import { fmtAmount, fmtPrice, executionSideLabel, executionSideClass } from '../../utils/format'

const props = defineProps({
  draftMeta: { type: Object, default: null },
  draftOrders: { type: Array, default: () => [] },
  initialExpanded: { type: Boolean, default: false },
})

const emit = defineEmits(['execution-complete', 'clear-draft'])

const config = ref(null)
const preview = ref(null)
const results = ref(null)
const history = ref([])
const error = ref('')
const previewLoading = ref(false)
const submitting = ref(false)
const expanded = ref(false)

const form = ref({
  brokerKind: 'paper',
  dryRun: true,
  maxOrderValue: '',
  maxPositionRatio: 1,
})

const previewOrders = computed(() => preview.value?.orders || [])
const runs = computed(() => history.value || [])

onMounted(async () => {
  await Promise.all([loadConfig(), loadHistory()])
  expanded.value = props.initialExpanded
  if (props.initialExpanded && props.draftOrders.length) {
    await previewExecution()
  }
})

async function loadConfig() {
  const data = await api('/api/execution/config')
  if (!data || data.error) return
  config.value = data
  form.value.brokerKind = data.defaultBrokerKind || 'paper'
  form.value.dryRun = data.defaultDryRun !== false
  form.value.maxPositionRatio = data.riskDefaults?.maxPositionRatio ?? 1
  form.value.maxOrderValue = data.riskDefaults?.maxOrderValue ?? ''
}

async function loadHistory() {
  const data = await api('/api/execution/history?limit=30')
  if (data?.runs) history.value = data.runs
}

function buildRiskPayload() {
  return {
    maxOrderValue: form.value.maxOrderValue === '' ? null : Number(form.value.maxOrderValue),
    maxPositionRatio: Number(form.value.maxPositionRatio || 1),
  }
}

async function previewExecution() {
  if (!props.draftOrders.length) return
  previewLoading.value = true
  error.value = ''
  results.value = null
  const data = await api('/api/execution/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trades: props.draftOrders, risk: buildRiskPayload() }),
  })
  if (!data || data.error) {
    error.value = data?.error || '生成执行预览失败'
    previewLoading.value = false
    return
  }
  preview.value = data
  previewLoading.value = false
}

async function submit() {
  if (!previewOrders.value.length) {
    error.value = '没有可提交的执行订单'
    return
  }
  submitting.value = true
  error.value = ''
  const data = await api('/api/execution/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      brokerKind: form.value.brokerKind,
      dryRun: form.value.dryRun,
      risk: buildRiskPayload(),
      confirm: true,
      orders: previewOrders.value,
    }),
  })
  submitting.value = false
  if (!data || data.error) {
    error.value = data?.error || '提交执行失败'
    return
  }
  results.value = data
  emit('execution-complete')
}

function clear() {
  emit('clear-draft')
  preview.value = null
  results.value = null
  error.value = ''
}
</script>

<template>
  <div class="bg-white rounded-xl border border-surface-3 p-4" data-testid="execution-workspace">
    <div class="flex flex-wrap items-center gap-3 justify-between">
      <div class="min-w-0">
        <div class="flex flex-wrap items-center gap-2">
          <h3 class="text-sm font-semibold text-slate-700">执行工作台</h3>
          <span class="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] text-amber-700" data-testid="execution-safety-banner">
            {{ form.brokerKind || 'paper' }} / {{ form.dryRun ? 'dry-run' : 'live' }}
          </span>
          <span v-if="draftMeta" class="rounded-full bg-brand-50 px-2 py-0.5 text-[11px] text-brand-700">
            已导入草案 · {{ previewOrders.length }} 笔
          </span>
          <span v-if="results?.summary" class="rounded-full bg-success/10 px-2 py-0.5 text-[11px] text-success">
            accepted {{ results.summary.accepted }}/{{ results.summary.total }}
          </span>
        </div>
        <p class="mt-1 text-xs text-slate-500">
          持仓信息优先展示；需要下单时再展开执行面板。默认模拟提交，不会默认真实下单。
        </p>
      </div>
      <button class="inline-flex items-center gap-1.5 rounded-lg border border-surface-3 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-surface-2 transition cursor-pointer" @click="expanded = !expanded" data-testid="execution-toggle">
        {{ expanded ? '收起执行面板' : (draftMeta ? '展开执行草案' : '展开执行面板') }}
        <svg :class="['w-4 h-4 text-slate-400 transition-transform', expanded ? 'rotate-180' : '']" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5"/>
        </svg>
      </button>
    </div>

    <div v-if="draftMeta && !expanded" class="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-xs text-brand-700" data-testid="execution-draft-meta">
      <span class="font-medium">Buffered 草案</span>
      <span>来源：{{ draftMeta.alias || '--' }}</span>
      <span>交易日：{{ draftMeta.tradeDate || '--' }}</span>
      <span>预计买入：{{ fmtAmount(draftMeta.summary?.estimatedBuyAmount) }}</span>
      <span>预计卖出：{{ fmtAmount(draftMeta.summary?.estimatedSellAmount) }}</span>
    </div>

    <div v-if="error && !expanded" class="mt-3 rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-sm text-danger" data-testid="execution-error">
      {{ error }}
    </div>

    <div v-if="expanded" class="mt-4 space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">Broker mode</label>
          <select v-model="form.brokerKind" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none cursor-pointer" data-testid="execution-broker-kind">
            <option v-for="item in config?.supportedBrokers || ['paper']" :key="item" :value="item">{{ item }}</option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">dry-run</label>
          <label class="flex items-center gap-2 px-3 py-2 rounded-lg border border-surface-3 bg-white cursor-pointer min-h-[42px]">
            <input v-model="form.dryRun" type="checkbox" class="accent-brand-600" data-testid="execution-dry-run" />
            <span class="text-sm text-slate-700">仅模拟提交</span>
          </label>
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">max order value</label>
          <input v-model="form.maxOrderValue" type="number" min="0" step="1000" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" data-testid="execution-max-order-value" />
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1.5">max position ratio</label>
          <input v-model="form.maxPositionRatio" type="number" min="0" max="1" step="0.01" class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none font-mono" data-testid="execution-max-position-ratio" />
        </div>
      </div>

      <div v-if="draftMeta" class="rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-700" data-testid="execution-draft-meta">
        <div class="font-medium">已导入 Buffered 调仓草案</div>
        <div class="mt-1 text-xs">来源：{{ draftMeta.alias || '--' }} · 交易日：{{ draftMeta.tradeDate || '--' }}</div>
        <div class="mt-2 flex flex-wrap gap-2 text-xs text-brand-700">
          <span>预计买入：{{ fmtAmount(draftMeta.summary?.estimatedBuyAmount) }}</span>
          <span>预计卖出：{{ fmtAmount(draftMeta.summary?.estimatedSellAmount) }}</span>
          <span>预计费用：{{ fmtAmount(draftMeta.summary?.estimatedFees) }}</span>
        </div>
      </div>

      <div v-if="error" class="rounded-xl border border-danger/20 bg-danger/5 px-4 py-3 text-sm text-danger" data-testid="execution-error">{{ error }}</div>

      <div class="flex flex-wrap gap-2">
        <button class="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" :disabled="previewLoading || !draftMeta" @click="previewExecution" data-testid="execution-preview-button">
          {{ previewLoading ? '生成中...' : '重新生成执行预览' }}
        </button>
        <button class="px-4 py-2 rounded-lg bg-cta text-white text-sm font-medium hover:bg-cta/90 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed" :disabled="submitting || !previewOrders.length" @click="submit" data-testid="execution-submit-button">
          {{ submitting ? '提交中...' : '确认模拟提交' }}
        </button>
        <button class="px-4 py-2 rounded-lg border border-surface-3 text-sm font-medium text-slate-600 hover:bg-surface-2 transition cursor-pointer" @click="clear">清空草案</button>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
          <div class="flex items-center justify-between mb-3">
            <h4 class="text-sm font-semibold text-slate-700">订单预览</h4>
            <div v-if="preview?.summary" class="text-xs text-slate-500">valid {{ preview.summary.validOrders }}/{{ preview.summary.totalOrders }}</div>
          </div>
          <div v-if="!previewOrders.length" class="text-sm text-slate-400 py-8 text-center" data-testid="execution-preview-empty">暂无执行预览</div>
          <div v-else class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">代码</th>
                  <th class="py-2 pr-3">方向</th>
                  <th class="py-2 pr-3 text-right">价格</th>
                  <th class="py-2 pr-3 text-right">数量</th>
                  <th class="py-2 pr-3 text-right">金额</th>
                  <th class="py-2 pr-3">校验</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in previewOrders" :key="`${row.stockId}-${row.side}`" class="border-b border-surface-3/50 last:border-0">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.stockId }}</td>
                  <td class="py-2.5 pr-3"><span :class="['inline-flex px-2 py-1 rounded-full text-[11px] font-medium', executionSideClass(row.side)]">{{ executionSideLabel(row.side) }}</span></td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ fmtPrice(row.price) }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ row.amount.toLocaleString() }}</td>
                  <td class="py-2.5 pr-3 text-right font-mono text-xs text-slate-700">{{ fmtAmount(row.orderValue) }}</td>
                  <td class="py-2.5 pr-3 text-xs">
                    <span v-if="row.valid" class="text-success">可提交</span>
                    <span v-else class="text-danger">{{ row.validationError }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
          <div class="flex items-center justify-between mb-3">
            <h4 class="text-sm font-semibold text-slate-700">执行结果</h4>
            <div v-if="results?.summary" class="text-xs text-slate-500">accepted {{ results.summary.accepted }}/{{ results.summary.total }}</div>
          </div>
          <div v-if="!results?.results?.length" class="text-sm text-slate-400 py-8 text-center" data-testid="execution-results-empty">暂无执行结果</div>
          <div v-else class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                  <th class="py-2 pr-3">代码</th>
                  <th class="py-2 pr-3">方向</th>
                  <th class="py-2 pr-3">状态</th>
                  <th class="py-2 pr-3">post-check</th>
                  <th class="py-2 pr-3">备注</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in results.results" :key="`${row.stockId}-${row.side}-${row.orderId || row.rejectionReason}`" class="border-b border-surface-3/50 last:border-0">
                  <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ row.stockId }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ executionSideLabel(row.side) }}</td>
                  <td class="py-2.5 pr-3 text-xs" :class="row.accepted ? 'text-success' : 'text-danger'">{{ row.accepted ? (row.status || 'accepted') : 'rejected' }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.postCheckStatus }}</td>
                  <td class="py-2.5 pr-3 text-xs text-slate-600">{{ row.rejectionReason || row.orderId || '--' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="bg-surface-1/40 rounded-xl border border-surface-3 p-4">
        <h4 class="text-sm font-semibold text-slate-700 mb-3">执行历史</h4>
        <div v-if="!runs.length" class="text-sm text-slate-400 py-6 text-center" data-testid="execution-history-empty">暂无执行历史</div>
        <div v-else class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
                <th class="py-2 pr-3">时间</th>
                <th class="py-2 pr-3">broker</th>
                <th class="py-2 pr-3">dry-run</th>
                <th class="py-2 pr-3 text-right">accepted</th>
                <th class="py-2 pr-3 text-right">rejected</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="run in runs" :key="run.historyId" class="border-b border-surface-3/50 last:border-0">
                <td class="py-2.5 pr-3 font-mono text-xs text-slate-700">{{ run.submittedAt }}</td>
                <td class="py-2.5 pr-3 text-xs text-slate-600">{{ run.brokerKind }}</td>
                <td class="py-2.5 pr-3 text-xs text-slate-600">{{ run.dryRun ? 'yes' : 'no' }}</td>
                <td class="py-2.5 pr-3 text-right font-mono text-xs text-success">{{ run.summary?.accepted ?? 0 }}</td>
                <td class="py-2.5 pr-3 text-right font-mono text-xs text-danger">{{ run.summary?.rejected ?? 0 }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>
