import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../utils/api'

export const useWatchlistStore = defineStore('watchlist', () => {
  const symbols = ref([])
  const loading = ref(false)
  const error = ref('')
  const pending = ref({})

  async function load() {
    loading.value = true
    error.value = ''
    try {
      const data = await api('/api/watchlist')
      symbols.value = Array.isArray(data?.symbols) ? data.symbols : []
    } catch {
      error.value = '自选列表加载失败'
    } finally {
      loading.value = false
    }
  }

  function isWatchlisted(symbol) {
    return symbols.value.includes(symbol)
  }

  async function toggle(symbol) {
    pending.value = { ...pending.value, [symbol]: true }
    try {
      const data = isWatchlisted(symbol)
        ? await api(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' })
        : await api('/api/watchlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol }),
          })
      symbols.value = Array.isArray(data?.symbols) ? data.symbols : symbols.value
    } catch {
      error.value = isWatchlisted(symbol) ? '移出自选失败' : '加入自选失败'
    } finally {
      pending.value = { ...pending.value, [symbol]: false }
    }
  }

  return { symbols, loading, error, pending, load, isWatchlisted, toggle }
})
