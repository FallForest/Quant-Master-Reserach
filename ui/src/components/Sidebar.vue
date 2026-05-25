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

const navItems = [
  { path: '/', label: '总览', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4' },
  { path: '/pipeline', label: '数据管道', icon: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15' },
  { path: '/browser', label: '数据浏览', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' },
  { path: '/factor',  label: '因子分析', icon: 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z' },
]

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
    <nav class="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
      <button v-for="item in navItems" :key="item.path"
              @click="navigate(item.path)"
              :class="[
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer text-left',
                isActive(item.path)
                  ? 'bg-white/10 text-white'
                  : 'text-brand-200 hover:bg-white/10 hover:text-white'
              ]">
        <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon"/>
        </svg>
        <span class="sidebar-label">{{ item.label }}</span>
      </button>
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
