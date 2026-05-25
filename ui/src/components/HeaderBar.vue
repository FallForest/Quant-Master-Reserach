<script setup>
defineProps({ title: String })
defineEmits(['toggle-mobile'])
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
      <div class="w-8 h-8 rounded-full bg-brand-100 text-brand-700 flex items-center justify-center text-sm font-semibold">Q</div>
    </div>
  </header>
</template>
