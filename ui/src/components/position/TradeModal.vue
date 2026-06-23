<script setup>
import { ref, watch } from 'vue'
import { api } from '../../utils/api'
import { fmtAmount } from '../../utils/format'

const emit = defineEmits(['close', 'saved'])

const props = defineProps({
  position: { type: Object, default: null },
  feeSettings: { type: Object, default: null },
})

const side = ref('buy')
const form = ref({ instrument: '', shares: '100', price: '' })
const saving = ref(false)
const error = ref('')
const fee = ref(null)
const maxShares = ref(0)

const QUICK_PCTS = [0.25, 0.5, 0.75, 1]
const PCT_LABELS = { 0.25: '1/4', 0.5: '1/2', 0.75: '3/4', 1: '全部' }

function setSharesByPct(pct) {
  if (side.value === 'sell' && maxShares.value > 0) {
    const rounded = Math.floor(maxShares.value * pct / 100) * 100
    form.value.shares = String(Math.max(rounded, 100))
    calcFee()
  }
}

function fillMarketPrice() {
  if (props.position?.currentPrice) {
    form.value.price = String(props.position.currentPrice)
    calcFee()
  }
}

/** Called by parent via template ref to initialize state before showing. */
function open(s) {
  side.value = s
  maxShares.value = (s === 'sell' && props.position) ? (props.position.shares || 0) : 0
  form.value = {
    instrument: props.position?.instrument || '',
    shares: s === 'sell' && props.position ? String(props.position.shares || 0) : '100',
    price: props.position?.currentPrice ? String(props.position.currentPrice) : '',
  }
  error.value = ''
  calcFee()
}

function calcFee() {
  const price = parseFloat(form.value.price)
  const shares = parseInt(form.value.shares, 10)
  if (!price || !shares || shares <= 0) { fee.value = null; return }
  const tradeValue = price * shares
  const fs = props.feeSettings || {}

  // Commission: rate with 5 CNY minimum for A-shares
  const commissionRate = parseFloat(fs.stockCommissionRate) || 0.00025
  let commission = tradeValue * commissionRate
  const MIN_COMMISSION = 5
  if (commission > 0 && commission < MIN_COMMISSION) commission = MIN_COMMISSION

  // Stamp duty: sell side only, default 0.05%
  const stampDutyRate = parseFloat(fs.stampDutyRate) || 0.0005
  const stampDuty = side.value === 'sell' ? tradeValue * stampDutyRate : 0

  // Transfer fee: SH stocks only, default 0.001%
  const transferFeeRate = parseFloat(fs.shTransferFeeRate) || 0.00001
  const isSH = form.value.instrument.toUpperCase().startsWith('SH')
  const transferFee = isSH ? tradeValue * transferFeeRate : 0

  const totalFees = commission + stampDuty + transferFee
  fee.value = {
    tradeValue: Math.round(tradeValue * 100) / 100,
    commission: Math.round(commission * 100) / 100,
    stampDuty: Math.round(stampDuty * 100) / 100,
    transferFee: Math.round(transferFee * 100) / 100,
    total: Math.round(totalFees * 100) / 100,
    netAmount: side.value === 'buy'
      ? Math.round((tradeValue + totalFees) * 100) / 100
      : Math.round((tradeValue - totalFees) * 100) / 100,
  }
}

watch(side, () => { error.value = ''; calcFee() })

async function submit() {
  const inst = form.value.instrument.trim().toUpperCase()
  const shares = parseInt(form.value.shares, 10)
  const price = parseFloat(form.value.price)

  if (!inst || inst.length < 6) { error.value = '请输入有效的股票代码'; return }
  if (!shares || shares <= 0) { error.value = '数量必须大于 0'; return }
  if (shares % 100 !== 0) { error.value = 'A 股数量必须为 100 的整数倍'; return }
  if (!price || price <= 0) { error.value = '价格必须大于 0'; return }

  if (side.value === 'sell' && maxShares.value > 0 && shares > maxShares.value) {
    error.value = `可卖数量不足，最多 ${maxShares.value} 股`
    return
  }

  saving.value = true
  error.value = ''
  try {
    const payload = { instrument: inst, shares, price: Math.round(price * 100) / 100 }
    const url = side.value === 'buy' ? '/api/positions' : '/api/positions/sell'
    const data = await api(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (data?.error) { error.value = data.error }
    else if (data) { emit('saved', data); emit('close') }
  } catch (e) { error.value = e.message || '提交失败' }
  finally { saving.value = false }
}

defineExpose({ open })
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center" @keydown.esc="$emit('close')">
      <div class="absolute inset-0 bg-black/40 backdrop-blur-sm" @click="$emit('close')"></div>
      <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6 animate-slide-in">
        <button @click="$emit('close')" class="absolute top-4 right-4 p-1 rounded-lg hover:bg-surface-2 text-slate-400 hover:text-slate-600 transition cursor-pointer" aria-label="关闭">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>

        <div class="flex items-center gap-3 mb-1">
          <h3 class="text-base font-semibold text-slate-800">快速交易</h3>
          <button @click="side = side === 'buy' ? 'sell' : 'buy'" :class="['inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold transition cursor-pointer border', side === 'buy' ? 'bg-bull/10 text-bull border-bull/20 hover:bg-bull/20' : 'bg-bear/10 text-bear border-bear/20 hover:bg-bear/20']">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h9.75m3.75-9l4.5-4.5M21 7.5l-4.5 4.5M21 7.5h-9.75"/></svg>
            {{ side === 'buy' ? '买入' : '卖出' }}
          </button>
        </div>
        <p class="text-xs text-slate-400 mb-5">点击方向标签切换买卖 · 回车或按钮提交</p>

        <div class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">股票代码</label>
            <input v-model="form.instrument" type="text" placeholder="SH600011" class="w-full px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" :disabled="saving" />
            <p class="text-[10px] text-slate-400 mt-1">格式：SH600000 或 SZ000001</p>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">价格（元）</label>
            <div class="flex items-center gap-2">
              <input v-model="form.price" type="text" inputmode="decimal" placeholder="8.50" class="flex-1 px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" @input="calcFee" @keydown.enter="submit" :disabled="saving" />
              <button v-if="position?.currentPrice" @click="fillMarketPrice" class="px-2 py-2.5 rounded-lg text-[11px] font-medium border border-surface-3 text-slate-500 hover:bg-surface-2 transition cursor-pointer whitespace-nowrap" :disabled="saving">
                现价
              </button>
            </div>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">数量（股）</label>
            <input v-model="form.shares" type="number" min="100" step="100" placeholder="100" class="w-full px-3 py-2.5 text-sm rounded-lg border border-surface-3 bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono" @input="calcFee" @keydown.enter="submit" :disabled="saving" />
            <div v-if="side === 'sell' && maxShares > 0" class="flex items-center gap-1.5 mt-2">
              <button v-for="pct in QUICK_PCTS" :key="pct" @click="setSharesByPct(pct)" :class="['px-2.5 py-1 rounded text-[11px] font-medium border transition cursor-pointer', parseInt(form.shares) === Math.floor(maxShares * pct / 100) * 100 ? 'bg-bear/10 text-bear border-bear/30' : 'border-surface-3 text-slate-500 hover:bg-surface-2']" :disabled="saving">
                {{ PCT_LABELS[pct] }}
              </button>
              <span class="text-[10px] text-slate-400 ml-1">可卖 {{ maxShares }} 股</span>
            </div>
          </div>
        </div>

        <div v-if="fee" class="mt-4 space-y-1.5 px-3 py-2.5 rounded-lg bg-surface-2/50 text-xs">
          <div class="flex justify-between text-slate-500">
            <span>交易金额</span>
            <span class="font-mono text-slate-700">{{ fmtAmount(fee.tradeValue) }}</span>
          </div>
          <div class="flex justify-between text-slate-500">
            <span>佣金</span>
            <span class="font-mono text-slate-700">{{ fmtAmount(fee.commission) }}</span>
          </div>
          <div v-if="fee.stampDuty > 0" class="flex justify-between text-slate-500">
            <span>印花税（卖出）</span>
            <span class="font-mono text-slate-700">{{ fmtAmount(fee.stampDuty) }}</span>
          </div>
          <div v-if="fee.transferFee > 0" class="flex justify-between text-slate-500">
            <span>过户费</span>
            <span class="font-mono text-slate-700">{{ fmtAmount(fee.transferFee) }}</span>
          </div>
          <div class="flex justify-between font-medium text-slate-700 pt-1 border-t border-surface-3">
            <span>{{ side === 'buy' ? '总支出（含费）' : '净收入（扣费）' }}</span>
            <span class="font-mono" :class="side === 'buy' ? 'text-bull' : 'text-bear'">{{ fmtAmount(fee.netAmount) }}</span>
          </div>
        </div>

        <div v-if="error" class="mt-3 flex items-center gap-2 text-xs text-danger bg-danger/5 rounded-lg px-3 py-2">
          <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
          </svg>
          {{ error }}
        </div>

        <div class="flex items-center gap-3 mt-5">
          <button @click="$emit('close')" class="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium border border-surface-3 text-slate-600 hover:bg-surface-2 transition cursor-pointer" :disabled="saving">取消</button>
          <button @click="submit" :disabled="saving" :class="['flex-1 flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer', saving ? 'bg-surface-2 text-slate-400 cursor-not-allowed' : side === 'buy' ? 'bg-bull text-white hover:bg-bull/90' : 'bg-bear text-white hover:bg-bear/90']">
            <svg v-if="saving" class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            {{ saving ? '提交中...' : (side === 'buy' ? '确认买入' : '确认卖出') }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
