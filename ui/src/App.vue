<script setup>
import { ref, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './components/Sidebar.vue'
import HeaderBar from './components/HeaderBar.vue'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileOpen = ref(false)

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
  </div>
</template>
