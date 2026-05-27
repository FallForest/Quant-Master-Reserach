<script setup>
import { ref, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './components/Sidebar.vue'
import HeaderBar from './components/HeaderBar.vue'
import { useToast } from './utils/toast'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileOpen = ref(false)
const { toasts, remove } = useToast()

const pageTitle = computed(() => route.meta?.title || '')

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
      <HeaderBar :title="pageTitle" @toggle-mobile="toggleMobile" />
      <div class="flex-1 overflow-y-auto">
        <router-view />
      </div>
    </div>

    <!-- Toast 通知 -->
    <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      <div v-for="t in toasts" :key="t.id"
           @click="remove(t.id)"
           :class="[
             'pointer-events-auto px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium cursor-pointer',
             'animate-slide-in max-w-xs',
             t.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
             t.type === 'error' ? 'bg-red-50 text-red-700 border border-red-200' :
             t.type === 'warn' ? 'bg-amber-50 text-amber-700 border border-amber-200' :
             'bg-white text-slate-700 border border-slate-200'
           ]">
        {{ t.msg }}
      </div>
    </div>
  </div>
</template>
