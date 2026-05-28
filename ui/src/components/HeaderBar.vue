<script setup>
defineProps({ title: String, isDark: Boolean })
defineEmits(['toggle-mobile', 'toggle-dark'])
</script>

<template>
  <header class="h-14 bg-white border-b border-surface-3 flex items-center justify-between px-4 sm:px-6 flex-shrink-0">
    <div class="flex items-center gap-3">
      <button @click="$emit('toggle-mobile')" class="md:hidden p-1.5 rounded-lg hover:bg-surface-2 transition cursor-pointer" aria-label="菜单">
        <svg class="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"/>
        </svg>
      </button>
      <h1 class="text-brand-900 font-semibold text-lg">{{ title }}</h1>
    </div>
    <div class="flex items-center gap-3">
      <span class="text-xs text-slate-400 font-mono hidden sm:inline">v2.1.0</span>
      <!-- Dark mode toggle -->
      <button @click="$emit('toggle-dark')"
        :aria-label="isDark ? '切换到浅色模式' : '切换到深色模式'"
        class="p-1.5 rounded-lg hover:bg-surface-2 transition cursor-pointer">
        <!-- Sun icon (shown in dark mode) -->
        <svg v-if="isDark" class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"/>
        </svg>
        <!-- Moon icon (shown in light mode) -->
        <svg v-else class="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z"/>
        </svg>
      </button>
      <div class="w-8 h-8 rounded-full bg-brand-100 text-brand-700 flex items-center justify-center text-sm font-semibold">Q</div>
    </div>
  </header>
</template>
