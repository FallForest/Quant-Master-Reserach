<script setup>
import { fmtAmount, fmtPrice } from '../../utils/format'

defineProps({
  orders: {
    type: Array,
    default: () => [],
  },
  expanded: {
    type: Boolean,
    default: false,
  },
})

defineEmits(['toggle'])

function orderSideLabel(side) {
  const map = { buy: '买入', sell: '卖出', close: '平仓' }
  return map[side] || side
}
</script>

<template>
  <div class="bg-white rounded-xl border border-surface-3 p-4">
    <button class="w-full flex items-center justify-between text-sm font-semibold text-slate-600 cursor-pointer" @click="$emit('toggle')">
      <div class="flex items-center gap-2">
        <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        历史委托
        <span class="text-[10px] font-normal text-slate-400">({{ orders.length }} 条)</span>
      </div>
      <svg :class="['w-4 h-4 text-slate-400 transition-transform', expanded ? 'rotate-180' : '']" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5"/>
      </svg>
    </button>
    <div v-if="expanded" class="mt-3 overflow-x-auto">
      <div v-if="!orders.length" class="text-center text-sm text-slate-400 py-6" data-testid="execution-history-empty">暂无委托记录</div>
      <table v-else class="w-full text-sm">
        <thead>
          <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
            <th class="py-2 pr-2">日期</th>
            <th class="py-2 pr-2">代码</th>
            <th class="py-2 pr-2">方向</th>
            <th class="py-2 pr-2 text-right">数量</th>
            <th class="py-2 pr-2 text-right">价格</th>
            <th class="py-2 pr-2 text-right">金额</th>
            <th class="py-2 pr-2 text-right">佣金</th>
            <th class="py-2 pr-2 text-right">印花税</th>
            <th class="py-2 pr-2 text-right">总费</th>
            <th class="py-2 pl-2">状态</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(o, i) in orders" :key="i" class="border-b border-surface-3/50 last:border-0">
            <td class="py-2 pr-2 font-mono text-xs text-slate-500">{{ o.date }}</td>
            <td class="py-2 pr-2 font-mono text-xs text-slate-700">{{ o.instrument || o.symbol }}</td>
            <td class="py-2 pr-2 text-xs">
              <span :class="['px-1.5 py-0.5 rounded text-[10px] font-medium', (o.side === 'buy' || o.side === 'open') ? 'bg-bull/10 text-bull' : 'bg-bear/10 text-bear']">{{ orderSideLabel(o.side) }}</span>
            </td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ (o.shares || o.amount || 0).toLocaleString() }}</td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtPrice(o.price) }}</td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtAmount((o.shares || o.amount || 0) * (o.price || 0)) }}</td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-500">{{ fmtAmount(o.feeBreakdown?.commission) }}</td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-500">{{ fmtAmount(o.feeBreakdown?.stampDuty) }}</td>
            <td class="py-2 pr-2 text-right font-mono text-xs text-slate-700 font-medium">{{ fmtAmount(o.feeBreakdown?.total) }}</td>
            <td class="py-2 pl-2 text-xs text-slate-500">{{ o.status || '已成交' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
