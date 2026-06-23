<script setup>
import { fmtAmount, fmtPct, pnlClass } from '../../utils/format'

const props = defineProps({
  positionData: {
    type: Object,
    default: null,
  },
})

const emit = defineEmits(['edit-cash'])
</script>

<template>
  <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
    <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
      <div class="text-[11px] text-slate-500 mb-1">总资产</div>
      <div class="text-xl font-bold font-mono text-slate-800">{{ fmtAmount(positionData?.totalAssets) }}</div>
      <div class="text-[10px] text-slate-400">持仓 + 现金</div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
      <div class="text-[11px] text-slate-500 mb-1">持仓市值</div>
      <div class="text-xl font-bold font-mono text-brand-600">{{ fmtAmount(positionData?.totalMarketValue) }}</div>
      <div class="text-[10px] text-slate-400">{{ positionData?.positionCount || 0 }} 只股票</div>
    </div>

    <div
      class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition cursor-pointer group"
      @click="$emit('edit-cash')"
    >
      <div class="flex items-center gap-1">
        <div class="text-[11px] text-slate-500 mb-1">可用现金</div>
        <svg class="w-3 h-3 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"/>
        </svg>
      </div>
      <div class="text-xl font-bold font-mono text-slate-800">{{ fmtAmount(positionData?.cash) }}</div>
      <div class="text-[10px] text-slate-400">点击编辑</div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
      <div class="text-[11px] text-slate-500 mb-1">总盈亏</div>
      <div :class="['text-xl font-bold font-mono', pnlClass(positionData?.totalPnl)]">
        {{ (positionData?.totalPnl || 0) > 0 ? '+' : '' }}{{ fmtAmount(positionData?.totalPnl) }}
      </div>
      <div :class="['text-[10px]', pnlClass(positionData?.totalPnlPct)]">
        {{ fmtPct(positionData?.totalPnlPct) }}
      </div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-3.5 hover:shadow-sm transition">
      <div class="text-[11px] text-slate-500 mb-1">仓位比例</div>
      <div class="text-xl font-bold font-mono text-slate-800">
        {{ positionData?.totalAssets > 0 ? ((positionData.totalMarketValue / positionData.totalAssets) * 100).toFixed(1) : 0 }}%
      </div>
      <div class="text-[10px] text-slate-400">股票 / 总资产</div>
    </div>
  </div>
</template>
