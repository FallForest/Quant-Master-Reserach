<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './components/Sidebar.vue'
import HeaderBar from './components/HeaderBar.vue'
import { useToast } from './utils/toast'
import { useDarkMode } from './utils/darkMode'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileOpen = ref(false)
const { toasts, remove } = useToast()
const { isDark, toggleDarkMode, initDarkMode } = useDarkMode()

const pageTitle = computed(() => route.meta?.title || '')

onMounted(() => {
  initDarkMode()
})

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function toggleMobile() {
  mobileOpen.value = !mobileOpen.value
}

function closeMobile() {
  mobileOpen.value = false
}

// 移动端路由切换时关闭侧边栏
watch(() => route.path, () => {
  if (window.innerWidth < 768) closeMobile()
})

const toastIcons = {
  success: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  error: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z',
  warn: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  info: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
}
</script>

<template>
  <div class="bg-surface-0 h-screen flex overflow-hidden">
    <!-- 移动端遮罩 -->
    <div v-if="mobileOpen" class="fixed inset-0 bg-black/30 z-30 md:hidden cursor-pointer" @click="closeMobile"></div>

    <Sidebar
      :collapsed="sidebarCollapsed"
      :mobileOpen="mobileOpen"
      @toggle="toggleSidebar"
      @close-mobile="closeMobile"
    />

    <div class="flex-1 flex flex-col min-w-0">
      <HeaderBar :title="pageTitle" :isDark="isDark" @toggle-mobile="toggleMobile" @toggle-dark="toggleDarkMode" />
      <div class="flex-1 overflow-y-auto">
        <router-view />
      </div>
    </div>

    <!-- Toast 通知 -->
    <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      <div v-for="t in toasts" :key="t.id"
           @click="remove(t.id)"
           role="alert"
           :class="[
             'pointer-events-auto px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium cursor-pointer',
             'animate-slide-in max-w-xs flex items-center gap-2',
             t.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
             t.type === 'error' ? 'bg-red-50 text-red-700 border border-red-200' :
             t.type === 'warn' ? 'bg-amber-50 text-amber-700 border border-amber-200' :
             'bg-white text-slate-700 border border-slate-200'
           ]">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" :d="toastIcons[t.type] || toastIcons.info"/>
        </svg>
        {{ t.msg }}
      </div>
    </div>
  </div>
</template>
