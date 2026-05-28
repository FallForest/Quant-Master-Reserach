import { ref, watch } from 'vue'

const isDark = ref(false)

// Initialize from localStorage or system preference
function initDarkMode() {
  const saved = localStorage.getItem('qm-dark-mode')
  if (saved !== null) {
    isDark.value = saved === 'true'
  } else {
    isDark.value = window.matchMedia('(prefers-color-scheme: dark)').matches
  }
  applyClass()
}

function applyClass() {
  document.documentElement.classList.toggle('dark', isDark.value)
  document.body.classList.toggle('dark', isDark.value)
}

function toggleDarkMode() {
  isDark.value = !isDark.value
  localStorage.setItem('qm-dark-mode', String(isDark.value))
  applyClass()
}

watch(isDark, () => applyClass())

export function useDarkMode() {
  return { isDark, toggleDarkMode, initDarkMode }
}
