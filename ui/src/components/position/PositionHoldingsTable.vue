<script setup>
import { fmtAmount, fmtPct, fmtPrice, pnlClass } from '../../utils/format'

const props = defineProps({
  positions: {
    type: Array,
    default: () => [],
  },
  sortKey: {
    type: String,
    required: true,
  },
  sortDir: {
    type: String,
    required: true,
  },
  selectedSymbol: {
    type: String,
    default: null,
  },
  deleteConfirm: {
    type: String,
    default: '',
  },
})

const emit = defineEmits([
  'toggle-sort',
  'select-position',
  'position-keydown',
  'edit-position',
  'delete-position',
  'trade-position',
])

function sortIcon(key) {
  if (props.sortKey !== key) return ''
  return props.sortDir === 'desc' ? ' ↓' : ' ↑'
}
</script>

<template>
  <div class="bg-white rounded-xl border border-surface-3 p-4">
    <div class="flex items-center gap-2 mb-3">
      <h3 class="text-sm font-semibold text-slate-600">持仓明细</h3>
      <span class="text-[10px] text-slate-400">点击行查看 K 线 · 行内编辑 · 点击删除需二次确认</span>
    </div>

    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
            <th class="py-2 pr-2 cursor-pointer select-none" @click="$emit('toggle-sort', 'instrument')">代码{{ sortIcon('instrument') }}</th>
            <th class="py-2 pr-2 cursor-pointer select-none" @click="$emit('toggle-sort', 'name')">名称{{ sortIcon('name') }}</th>
            <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="$emit('toggle-sort', 'shares')">持仓{{ sortIcon('shares') }}</th>
            <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="$emit('toggle-sort', 'costPrice')">成本{{ sortIcon('costPrice') }}</th>
            <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="$emit('toggle-sort', 'currentPrice')">现价{{ sortIcon('currentPrice') }}</th>
            <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="$emit('toggle-sort', 'marketValue')">市值{{ sortIcon('marketValue') }}</th>
            <th class="py-2 pr-2 text-right cursor-pointer select-none" @click="$emit('toggle-sort', 'pnlPct')">盈亏%{{ sortIcon('pnlPct') }}</th>
            <th class="py-2 pl-2 cursor-pointer select-none" @click="$emit('toggle-sort', 'weight')">占比{{ sortIcon('weight') }}</th>
            <th class="py-2 pl-2 text-right cursor-pointer select-none whitespace-nowrap" @click="$emit('toggle-sort', 'buyFee')">买费{{ sortIcon('buyFee') }}</th>
            <th class="py-2 pl-2 w-20 text-right">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="p in positions"
            :key="p.instrument"
            tabindex="0"
            :aria-label="`查看 ${p.name || p.instrument} K线`"
            :class="[
              'relative border-b border-surface-3/50 last:border-0 cursor-pointer transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-brand-400/40 focus:ring-inset',
              selectedSymbol === p.instrument ? 'bg-brand-50/80 hover:bg-brand-50' : 'hover:bg-surface-2/30',
            ]"
            @click="$emit('select-position', p)"
            @keydown="$emit('position-keydown', $event, p)"
            data-testid="position-row"
          >
            <td class="py-2.5 pr-2 font-mono text-xs text-slate-700 relative">
              <div v-if="selectedSymbol === p.instrument" class="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-brand-500"></div>
              <span class="pl-2">{{ p.instrument }}</span>
            </td>
            <td class="py-2.5 pr-2 text-xs text-slate-600">{{ p.name }}</td>
            <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ p.shares.toLocaleString() }}</td>
            <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-500">{{ fmtPrice(p.costPrice) }}</td>
            <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtPrice(p.currentPrice) }}</td>
            <td class="py-2.5 pr-2 text-right font-mono text-xs text-slate-700">{{ fmtAmount(p.marketValue) }}</td>
            <td :class="['py-2.5 pr-2 text-right font-mono text-xs', pnlClass(p.pnlPct)]">{{ fmtPct(p.pnlPct) }}</td>
            <td class="py-2.5 pl-2">
              <div class="flex items-center gap-1.5">
                <div class="flex-1 bg-surface-2 rounded-full h-1.5">
                  <div class="h-1.5 rounded-full bg-brand-500/70 transition-all duration-300" :style="{ width: Math.min(p.weight, 100) + '%' }"></div>
                </div>
                <span class="text-[10px] font-mono text-slate-400 w-8 text-right">{{ p.weight }}%</span>
              </div>
            </td>
            <td class="py-2.5 pl-2 text-right font-mono text-xs text-slate-600 cursor-help"
                :title="`佣金: ${fmtAmount(p.buyFee?.commission)} | 印花税: ${fmtAmount(p.buyFee?.stampDuty)} | 过户费: ${fmtAmount(p.buyFee?.transferFee)}`">
              {{ fmtAmount(p.buyFee?.total) }}
            </td>
            <td class="py-2.5 pl-2">
              <div class="flex items-center justify-end gap-1">
                <button @click.stop="$emit('trade-position', p)" class="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border border-surface-3 text-slate-600 hover:bg-surface-2 transition cursor-pointer">
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h9.75m3.75-9l4.5-4.5M21 7.5l-4.5 4.5M21 7.5h-9.75"/></svg>
                  交易
                </button>
                <button @click.stop="$emit('edit-position', p)" class="p-1 rounded hover:bg-brand-50 text-slate-400 hover:text-brand-600 transition cursor-pointer" aria-label="编辑">
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"/>
                  </svg>
                </button>
                <button
                  @click.stop="$emit('delete-position', p.instrument)"
                  :class="['p-1 rounded transition cursor-pointer', deleteConfirm === p.instrument ? 'bg-danger/10 text-danger' : 'hover:bg-danger/5 text-slate-400 hover:text-danger']"
                  :aria-label="deleteConfirm === p.instrument ? '确认删除' : '删除'"
                >
                  <svg v-if="deleteConfirm === p.instrument" class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"/>
                  </svg>
                  <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/>
                  </svg>
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
