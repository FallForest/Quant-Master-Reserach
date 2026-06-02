<script setup>
import { computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import BufferedRebalance from './BufferedRebalance.vue'

const route = useRoute()
const router = useRouter()

const strategies = [
  {
    id: 'buffered-rebalance',
    label: 'Buffered 调仓',
    route: '/strategy/buffered-rebalance',
    status: '已接入',
    description: '保留 buffer 范围内旧仓位，并按目标权重补齐仓位，减少不必要换手。',
    component: BufferedRebalance,
  },
]

const strategyMap = new Map(strategies.map((strategy) => [strategy.route, strategy]))

const activeStrategy = computed(() => {
  if (route.path === '/strategy') return strategies[0]
  return strategyMap.get(route.path) || strategies[0]
})

watch(
  () => route.path,
  (path) => {
    if (path === '/strategy') {
      router.replace(strategies[0].route)
    }
  },
  { immediate: true }
)

function selectStrategy(strategy) {
  if (route.path !== strategy.route) {
    router.push(strategy.route)
  }
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">
    <div class="flex flex-wrap items-start gap-3">
      <div>
        <h2 class="text-base font-semibold text-slate-700">策略调仓</h2>
        <p class="text-xs text-slate-500 mt-1">这里作为调仓策略总入口，当前先接入 Buffered 调仓，后续可继续扩展更多策略实现。</p>
      </div>
      <div class="flex-1"></div>
      <div class="inline-flex items-center rounded-full bg-brand-50 px-3 py-1 text-[11px] font-medium text-brand-700 border border-brand-200">
        当前已接入 {{ strategies.length }} 个策略
      </div>
    </div>

    <div class="bg-white rounded-xl border border-surface-3 p-4 space-y-4">
      <div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div class="text-sm font-semibold text-slate-700">策略目录</div>
          <p class="text-xs text-slate-500 mt-1">先选择调仓策略，再查看该策略的解释、参数与调仓结果。</p>
        </div>
        <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5 overflow-x-auto">
          <button
            v-for="strategy in strategies"
            :key="strategy.id"
            :aria-label="strategy.label"
            :aria-current="activeStrategy.route === strategy.route ? 'page' : undefined"
            :class="[
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-100 whitespace-nowrap',
              activeStrategy.route === strategy.route
                ? 'bg-white text-brand-600 shadow-sm'
                : 'text-slate-500 hover:text-slate-700',
            ]"
            @click="selectStrategy(strategy)"
          >
            {{ strategy.label }}
          </button>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)] gap-4">
        <div class="rounded-xl border border-surface-3 bg-surface-1/40 px-4 py-4">
          <div class="flex items-center gap-2 mb-2">
            <div class="text-sm font-semibold text-slate-700">{{ activeStrategy.label }}</div>
            <span class="inline-flex px-2 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-medium">
              {{ activeStrategy.status }}
            </span>
          </div>
          <p class="text-sm text-slate-600 leading-6">{{ activeStrategy.description }}</p>
        </div>
        <div class="rounded-xl border border-surface-3 bg-white px-4 py-4">
          <div class="text-xs text-slate-500 mb-2">扩展建议</div>
          <ul class="space-y-2 text-sm text-slate-600 list-disc pl-4">
            <li>后续新增策略时，只需补充策略元信息和对应组件。</li>
            <li>容器页保持统一入口，侧边栏与总览快捷操作无需频繁改动。</li>
            <li>未来若策略增多，可再升级为更完整的策略目录或筛选面板。</li>
          </ul>
        </div>
      </div>
    </div>

    <component :is="activeStrategy.component" embedded />
  </div>
</template>
