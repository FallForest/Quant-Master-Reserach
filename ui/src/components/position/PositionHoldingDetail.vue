<script setup>
import KlineContent from '../KlineContent.vue'

defineProps({
  symbol: {
    type: String,
    default: null,
  },
  name: {
    type: String,
    default: '',
  },
  quote: {
    type: Object,
    default: () => ({}),
  },
  period: {
    type: String,
    default: 'D',
  },
  loadingMin: {
    type: Boolean,
    default: false,
  },
  marketOpen: {
    type: Boolean,
    default: false,
  },
  chartId: {
    type: String,
    default: 'position-kline-chart',
  },
})

defineEmits(['close', 'set-period'])
</script>

<template>
  <div class="rounded-xl border border-surface-3 overflow-hidden min-h-[520px] bg-white flex flex-col" data-testid="position-kline-panel">
    <KlineContent
      :symbol="symbol"
      :name="name"
      :quote="quote"
      :period="period"
      :loading-min="loadingMin"
      :market-open="marketOpen"
      :chart-id="chartId"
      @close="$emit('close')"
      @set-period="$emit('set-period', $event)"
    />
  </div>
</template>
