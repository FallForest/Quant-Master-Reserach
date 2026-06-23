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
    icon: 'M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5',
    accent: 'from-brand-500 to-blue-600',
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
    <!-- Header -->
    <div class="flex flex-wrap items-start gap-3">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-blue-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
          </svg>
        </div>
        <div>
          <h2 class="text-base font-semibold text-slate-800 dark:text-slate-100">策略调仓</h2>
          <p class="text-xs text-slate-500 mt-0.5">统一入口 · 调仓策略管理与执行</p>
        </div>
      </div>
      <div class="flex-1"></div>
      <div class="inline-flex items-center gap-1.5 rounded-full bg-brand-50 px-3 py-1.5 text-[11px] font-semibold text-brand-700 border border-brand-200/60">
        <span class="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse"></span>
        {{ strategies.length }} 个策略已接入
      </div>
    </div>

    <!-- Strategy Directory Card -->
    <div class="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-surface-3/80 dark:border-slate-700/50 p-5 space-y-5 shadow-sm">
      <div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div class="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2">
            <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
            </svg>
            策略目录
          </div>
          <p class="text-xs text-slate-500 mt-1 ml-6">选择调仓策略，查看策略解释、参数与调仓结果。</p>
        </div>
        <div class="flex items-center bg-surface-2/80 dark:bg-slate-700/50 rounded-xl p-0.5 gap-0.5 overflow-x-auto">
          <button
            v-for="strategy in strategies"
            :key="strategy.id"
            :aria-label="strategy.label"
            :aria-current="activeStrategy.route === strategy.route ? 'page' : undefined"
            :class="[
              'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer focus:outline-none focus:ring-2 focus:ring-brand-400/40 whitespace-nowrap',
              activeStrategy.route === strategy.route
                ? 'bg-white dark:bg-slate-600 text-brand-600 dark:text-brand-400 shadow-sm shadow-brand-500/10'
                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300',
            ]"
            @click="selectStrategy(strategy)"
          >
            {{ strategy.label }}
          </button>
        </div>
      </div>

      <!-- Strategy Info Grid -->
      <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)] gap-4">
        <div class="rounded-xl border border-brand-100/60 dark:border-brand-900/30 bg-gradient-to-br from-brand-50/50 to-blue-50/30 dark:from-brand-950/20 dark:to-blue-950/10 px-5 py-4">
          <div class="flex items-center gap-2 mb-2">
            <div class="text-sm font-semibold text-slate-700 dark:text-slate-200">{{ activeStrategy.label }}</div>
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[10px] font-semibold">
              <span class="w-1 h-1 rounded-full bg-emerald-500"></span>
              {{ activeStrategy.status }}
            </span>
          </div>
          <p class="text-sm text-slate-600 dark:text-slate-400 leading-6">{{ activeStrategy.description }}</p>
        </div>
        <div class="rounded-xl border border-surface-3/80 dark:border-slate-700/50 bg-white/60 dark:bg-slate-800/30 px-5 py-4">
          <div class="text-xs text-slate-500 font-semibold mb-3 flex items-center gap-1.5">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
            </svg>
            扩展建议
          </div>
          <ul class="space-y-2 text-xs text-slate-600 dark:text-slate-400">
            <li class="flex items-start gap-2">
              <span class="w-1 h-1 rounded-full bg-brand-400 mt-1.5 flex-shrink-0"></span>
              新增策略时，只需补充策略元信息和对应组件
            </li>
            <li class="flex items-start gap-2">
              <span class="w-1 h-1 rounded-full bg-brand-400 mt-1.5 flex-shrink-0"></span>
              容器页保持统一入口，侧边栏无需频繁改动
            </li>
            <li class="flex items-start gap-2">
              <span class="w-1 h-1 rounded-full bg-brand-400 mt-1.5 flex-shrink-0"></span>
              策略增多后可升级为完整目录或筛选面板
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Active Strategy Component -->
    <component :is="activeStrategy.component" embedded />
  </div>
</template>
