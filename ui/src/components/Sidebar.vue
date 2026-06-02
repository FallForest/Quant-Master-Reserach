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
    label: '总览',
    items: [
      { path: '/', label: '总览', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4' },
    ],
  },
  {
    label: '数据',
    items: [
      { path: '/browser', label: '数据浏览', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' },
    ],
  },
  {
    label: '策略',
    items: [
      { path: '/model', label: '模型选股', icon: 'M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6' },
      { path: '/strategy', label: '策略调仓', icon: 'M3.75 6.75h16.5M3.75 12h10.5m-10.5 5.25h16.5m-6-6l3 3m0 0l-3 3m3-3H9.75' },
    ],
  },
  {
    label: '执行',
    items: [
      { path: '/execution', label: '交易执行', icon: 'M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3' },
    ],
  },
]

const sidebarClass = computed(() => {
  const base = 'sidebar fixed md:relative z-40 h-full bg-brand-950 text-white flex flex-col shadow-xl md:shadow-none'
  const width = props.collapsed ? 'sidebar-collapsed' : 'sidebar-expanded'
  const mobile = props.mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
  return `${base} ${width} ${mobile}`
})

function isActive(path) {
  if (path === '/strategy') {
    return route.path === '/strategy' || route.path.startsWith('/strategy/')
  }
  return route.path === path
}

function navigate(path) {
  router.push(path)
  emit('close-mobile')
}
</script>

<template>
  <aside :class="sidebarClass">
    <div class="h-14 flex items-center gap-3 px-4 border-b border-white/10 flex-shrink-0">
      <svg class="w-7 h-7 text-cta flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/>
      </svg>
      <span class="sidebar-label font-semibold text-lg tracking-tight">QuantMaster</span>
    </div>

    <nav class="flex-1 py-3 px-2 space-y-4 overflow-y-auto">
      <div v-for="group in navGroups" :key="group.label">
        <div class="sidebar-label px-3 mb-1 text-[10px] font-semibold uppercase tracking-wider text-brand-400/60">{{ group.label }}</div>
        <div class="space-y-0.5">
          <button
            v-for="item in group.items"
            :key="item.path"
            :aria-current="isActive(item.path) ? 'page' : undefined"
            :aria-label="item.label"
            :class="[
              'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition cursor-pointer text-left',
              isActive(item.path)
                ? 'bg-white/10 text-white'
                : 'text-brand-200 hover:bg-white/10 hover:text-white',
            ]"
            @click="navigate(item.path)"
          >
            <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon"/>
            </svg>
            <span class="sidebar-label">{{ item.label }}</span>
          </button>
        </div>
      </div>
    </nav>

    <button
      aria-label="收起侧边栏"
      class="hidden md:flex items-center gap-3 px-3 py-3 border-t border-white/10 text-brand-400 hover:text-white text-sm cursor-pointer transition focus:outline-none"
      @click="emit('toggle')"
    >
      <svg
        :class="['w-5 h-5 flex-shrink-0 transition-transform', collapsed ? 'rotate-180' : '']"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        viewBox="0 0 24 24"
      >
        <path stroke-linecap="round" stroke-linejoin="round" d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5"/>
      </svg>
      <span class="sidebar-label">收起</span>
    </button>
  </aside>
</template>
