import { onUnmounted } from 'vue'

/**
 * Manages setInterval lifetimes scoped to a component lifecycle.
 * Returns helpers to create and clear intervals without manual cleanup.
 */
export function useManagedInterval() {
  const timers = new Set()

  function setManagedInterval(fn, ms) {
    const id = setInterval(fn, ms)
    timers.add(id)
    return id
  }

  function clearManagedTimers() {
    for (const id of timers) clearInterval(id)
    timers.clear()
  }

  onUnmounted(() => {
    clearManagedTimers()
  })

  return {
    timers,
    setManagedInterval,
    clearManagedTimers,
  }
}
