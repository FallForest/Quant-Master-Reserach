<script setup>
import { computed, ref, watch } from 'vue'
import { api } from '../../utils/api'
import { fmtAmount } from '../../utils/format'

const props = defineProps({
  feeSettings: { type: Object, default: null },
  capitalAmount: { type: Number, default: 0 },
})

const emit = defineEmits(['close', 'saved'])

const form = ref({
  capitalAmount: 0,
  stockCommissionRate: '',
  etfCommissionRate: '',
  stampDutyRate: '',
  shTransferFeeRate: '',
})

const saving = ref(false)
const error = ref('')
const success = ref(false)

watch(
  () => props.feeSettings,
  (s) => {
    if (s) {
      form.value.stockCommissionRate = String(s.stockCommissionRate ?? 0.0001)
      form.value.etfCommissionRate = String(s.etfCommissionRate ?? 0.00005)
      form.value.stampDutyRate = String(s.stampDutyRate ?? 0.0005)
      form.value.shTransferFeeRate = String(s.shTransferFeeRate ?? 0.00001)
    }
  },
  { immediate: true },
)

watch(
  () => props.capitalAmount,
  (v) => {
    form.value.capitalAmount = v || 0
  },
  { immediate: true },
)

function fmtRate(val) {
  const n = parseFloat(val)
  if (isNaN(n)) return '--'
  return (n * 10000).toFixed(1) + ' /万'
}

function fmtRateWan(val) {
  const n = parseFloat(val)
  if (isNaN(n) || n <= 0) return ''
  const v = n * 10000
  return v === Math.floor(v) ? `万${v}` : `万${v.toFixed(1)}`
}

function fmtRateShiwan(val) {
  const n = parseFloat(val)
  if (isNaN(n) || n <= 0) return ''
  const v = n * 100000
  return v === Math.floor(v) ? `十万${v}` : `十万${v.toFixed(1)}`
}

const ruleSummary = computed(() => {
  const s = parseFloat(form.value.stockCommissionRate)
  const e = parseFloat(form.value.etfCommissionRate)
  const st = parseFloat(form.value.stampDutyRate)
  const sh = parseFloat(form.value.shTransferFeeRate)
  const parts = []
  if (s > 0) parts.push(`股票${fmtRateWan(s)}`)
  if (e > 0) parts.push(`ETF${fmtRateWan(e)}`)
  if (st > 0) parts.push(`印花税买入不收，卖出${fmtRateWan(st)}`)
  if (sh > 0) parts.push(`沪市过户费买卖${fmtRateShiwan(sh)}，深市包含在佣金里`)
  return parts.join('；')
})

async function save() {
  saving.value = true
  error.value = ''
  success.value = false

  const body = {
    capitalAmount: parseFloat(form.value.capitalAmount) || 0,
  }
  for (const key of ['stockCommissionRate', 'etfCommissionRate', 'stampDutyRate', 'shTransferFeeRate']) {
    const v = parseFloat(form.value[key])
    if (!isNaN(v) && v >= 0) body[key] = v
  }

  const data = await api('/api/positions/account', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  saving.value = false
  if (data?.error) {
    error.value = data.error
    return
  }
  success.value = true
  emit('saved', data)
}
</script>

<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center">
    <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" @click="$emit('close')"></div>
    <div class="relative bg-white shadow-2xl w-full max-w-lg mx-4 animate-slide-in"
         style="border-top: 4px solid #002FA7; border-radius: 0;">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 pt-6 pb-4 border-b border-slate-200">
        <div>
          <h3 class="text-sm font-semibold uppercase tracking-widest text-[#002FA7]">账户费率设置</h3>
          <p class="text-xs text-slate-400 mt-1">Account Fee Configuration</p>
        </div>
        <button @click="$emit('close')" class="p-1 hover:bg-slate-100 transition cursor-pointer" aria-label="关闭">
          <svg class="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>

      <!-- Body -->
      <div class="px-6 py-5 space-y-5">
        <!-- Capital -->
        <div>
          <label class="block text-xs font-medium text-slate-600 uppercase tracking-wider mb-1.5">账户本金</label>
          <input v-model="form.capitalAmount" type="number" min="0" step="10000"
                 class="w-full px-3 py-2.5 text-sm border border-slate-300 bg-white focus:border-[#002FA7] outline-none transition font-mono"
                 style="border-radius: 0;" />
        </div>

        <!-- Fee rates grid -->
        <div>
          <div class="text-xs font-medium text-slate-600 uppercase tracking-wider mb-3">费率配置</div>
          <div class="grid grid-cols-2 gap-x-4 gap-y-4">
            <div>
              <label class="block text-[11px] text-slate-500 mb-1">股票佣金</label>
              <div class="flex items-center gap-2">
                <input v-model="form.stockCommissionRate" type="number" min="0" step="0.00001"
                       class="flex-1 px-2.5 py-2 text-sm border border-slate-300 bg-white focus:border-[#002FA7] outline-none transition font-mono"
                       style="border-radius: 0;" />
                <span class="text-[11px] text-slate-400 font-mono min-w-[4rem] text-right">{{ fmtRate(form.stockCommissionRate) }}</span>
              </div>
            </div>

            <div>
              <label class="block text-[11px] text-slate-500 mb-1">ETF 佣金</label>
              <div class="flex items-center gap-2">
                <input v-model="form.etfCommissionRate" type="number" min="0" step="0.00001"
                       class="flex-1 px-2.5 py-2 text-sm border border-slate-300 bg-white focus:border-[#002FA7] outline-none transition font-mono"
                       style="border-radius: 0;" />
                <span class="text-[11px] text-slate-400 font-mono min-w-[4rem] text-right">{{ fmtRate(form.etfCommissionRate) }}</span>
              </div>
            </div>

            <div>
              <label class="block text-[11px] text-slate-500 mb-1">印花税（卖出）</label>
              <div class="flex items-center gap-2">
                <input v-model="form.stampDutyRate" type="number" min="0" step="0.00001"
                       class="flex-1 px-2.5 py-2 text-sm border border-slate-300 bg-white focus:border-[#002FA7] outline-none transition font-mono"
                       style="border-radius: 0;" />
                <span class="text-[11px] text-slate-400 font-mono min-w-[4rem] text-right">{{ fmtRate(form.stampDutyRate) }}</span>
              </div>
            </div>

            <div>
              <label class="block text-[11px] text-slate-500 mb-1">沪市过户费</label>
              <div class="flex items-center gap-2">
                <input v-model="form.shTransferFeeRate" type="number" min="0" step="0.00001"
                       class="flex-1 px-2.5 py-2 text-sm border border-slate-300 bg-white focus:border-[#002FA7] outline-none transition font-mono"
                       style="border-radius: 0;" />
                <span class="text-[11px] text-slate-400 font-mono min-w-[4rem] text-right">{{ fmtRate(form.shTransferFeeRate) }}</span>
              </div>
            </div>

          </div>
        </div>

        <!-- Live fee rule summary -->
        <div class="px-3 py-3 border border-[#002FA7]/15 bg-[#002FA7]/5" style="border-left: 3px solid #002FA7;">
          <div class="text-[11px] font-medium text-[#002FA7] uppercase tracking-wider mb-1.5">费率规则预览</div>
          <p class="text-[12px] text-slate-700 leading-relaxed">{{ ruleSummary || '—' }}</p>
        </div>

        <!-- Status messages -->
        <div v-if="error" class="flex items-center gap-2 text-xs text-red-600 bg-red-50 px-3 py-2" style="border-left: 3px solid #dc2626;">
          {{ error }}
        </div>
        <div v-if="success" class="flex items-center gap-2 text-xs text-green-700 bg-green-50 px-3 py-2" style="border-left: 3px solid #16a34a;">
          费率设置已保存
        </div>
      </div>

      <!-- Footer -->
      <div class="flex items-center gap-3 px-6 py-4 border-t border-slate-200 bg-slate-50">
        <button @click="$emit('close')"
                class="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200 transition cursor-pointer"
                style="border-radius: 0;">
          取消
        </button>
        <div class="flex-1"></div>
        <button @click="save" :disabled="saving"
                class="px-6 py-2 text-sm font-semibold text-white transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                :style="{ backgroundColor: saving ? '#94A3B8' : '#002FA7', borderRadius: 0 }">
          {{ saving ? '保存中...' : '保存设置' }}
        </button>
      </div>
    </div>
  </div>
</template>
