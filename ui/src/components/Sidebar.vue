<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const props = defineProps({
  collapsed: Boolean,
  mobileOpen: Boolean,
})

const emit = defineEmits(['toggle', 'close-mobile'])

const route = useRoute()
const router = useRouter()

const navGroups = [
  {
    label: '概览',
    items: [
      { path: '/', label: '总览', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4' },
    ],
  },
  {
    label: '数据',
    items: [
      { path: '/pipeline', label: '数据管道', icon: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15' },
      { path: '/browser', label: '数据浏览', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' },
      { path: '/factor', label: '因子分析', icon: 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z' },
    ],
  },
  {
    label: '模型',
    items: [
      { path: '/model-lab', label: '模型工坊', icon: 'M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9' },
      { path: '/model-performance', label: '模型绩效', icon: 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z' },
      { path: '/stock-select', label: '模型选股', icon: 'M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6' },
      { path: '/experiments', label: '实验管理', icon: 'M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z' },
    ],
  },
  {
    label: '策略',
    items: [
      { path: '/strategy-lab', label: '策略工坊', icon: 'M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z' },
      { path: '/backtest', label: '策略回测', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
    ],
  },
  {
    label: '分析',
    items: [
      { path: '/portfolio', label: '持仓分析', icon: 'M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3' },
      { path: '/optimizer', label: '组合优化', icon: 'M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z' },
      { path: '/attribution', label: '收益归因', icon: 'M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6zM13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z' },
    ],
  },
]

const navItems = navGroups.flatMap(g => g.items)

const sidebarClass = computed(() => {
  const base = 'sidebar fixed md:relative z-40 h-full bg-brand-950 text-white flex flex-col shadow-xl md:shadow-none'
  const width = props.collapsed ? 'sidebar-collapsed' : 'sidebar-expanded'
  const mobile = props.mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
  return `${base} ${width} ${mobile}`
})

function isActive(path) {
  return route.path === path
}

function navigate(path) {
  router.push(path)
  emit('close-mobile')
}
</script>

<template>
  <aside :class="sidebarClass">
    <!-- Logo -->
    <div class="h-14 flex items-center gap-3 px-4 border-b border-white/10 flex-shrink-0">
      <svg class="w-7 h-7 text-cta flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/>
      </svg>
      <span class="sidebar-label font-semibold text-lg tracking-tight">QuantMaster</span>
    </div>

    <!-- Nav -->
    <nav class="flex-1 py-3 px-2 space-y-4 overflow-y-auto">
      <div v-for="group in navGroups" :key="group.label">
        <div class="sidebar-label px-3 mb-1 text-[10px] font-semibold uppercase tracking-wider text-brand-400/60">{{ group.label }}</div>
        <div class="space-y-0.5">
          <button v-for="item in group.items" :key="item.path"
                  @click="navigate(item.path)"
                  :aria-label="item.label"
                  :aria-current="isActive(item.path) ? 'page' : undefined"
                  :class="[
                    'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition cursor-pointer text-left',
                    isActive(item.path)
                      ? 'bg-white/10 text-white'
                      : 'text-brand-200 hover:bg-white/10 hover:text-white'
                  ]">
            <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon"/>
            </svg>
            <span class="sidebar-label">{{ item.label }}</span>
          </button>
        </div>
      </div>
    </nav>

    <!-- 折叠按钮 -->
    <button @click="emit('toggle')" aria-label="收起侧边栏"
      class="hidden md:flex items-center gap-3 px-3 py-3 border-t border-white/10 text-brand-400 hover:text-white text-sm cursor-pointer transition focus:outline-none">
      <svg :class="['w-5 h-5 flex-shrink-0 transition-transform', collapsed ? 'rotate-180' : '']"
           fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5"/>
      </svg>
      <span class="sidebar-label">收起</span>
    </button>
  </aside>
</template>
