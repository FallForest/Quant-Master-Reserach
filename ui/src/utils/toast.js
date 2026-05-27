import { ref } from 'vue'

const toasts = ref([])
let idCounter = 0

export function useToast() {
  function show(msg, type = 'info', duration = 3000) {
    // 去重：相同消息 30 秒内不重复弹出
    const now = Date.now()
    const dup = toasts.value.find(t => t.msg === msg && now - t.createdAt < 30000)
    if (dup) return

    const id = ++idCounter
    const toast = { id, msg, type, createdAt: now }
    toasts.value.push(toast)
    setTimeout(() => remove(id), duration)
  }

  function success(msg, duration) { show(msg, 'success', duration) }
  function error(msg, duration) { show(msg, 'error', duration) }
  function warn(msg, duration) { show(msg, 'warn', duration) }
  function info(msg, duration) { show(msg, 'info', duration) }

  function remove(id) {
    toasts.value = toasts.value.filter(t => t.id !== id)
  }

  return { toasts, show, success, error, warn, info, remove }
}
