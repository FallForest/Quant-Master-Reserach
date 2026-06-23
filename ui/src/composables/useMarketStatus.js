import { ref } from 'vue'
import { useManagedInterval } from './useManagedInterval'

/**
 * Tracks whether the Shanghai A-share market is currently open.
 * Polls every 10 seconds and cleans up on unmount (via useManagedInterval).
 */
export function useMarketStatus() {
  const marketOpen = ref(false)
  const { setManagedInterval, clearManagedTimers } = useManagedInterval()

  function checkMarketStatus() {
    const now = new Date()
    const shanghai = new Date(now.getTime() + 8 * 3600 * 1000)
    const minutes = shanghai.getUTCHours() * 60 + shanghai.getUTCMinutes()
    const dayOfWeek = shanghai.getUTCDay()
    const weekday = dayOfWeek >= 1 && dayOfWeek <= 5
    const morning = minutes >= 570 && minutes < 690 // 9:30-11:30
    const afternoon = minutes >= 780 && minutes < 900 // 13:00-15:00
    marketOpen.value = weekday && (morning || afternoon)
  }

  function startMarketCheck() {
    checkMarketStatus()
    setManagedInterval(checkMarketStatus, 10000)
  }

  return {
    marketOpen,
    checkMarketStatus,
    startMarketCheck,
    clearManagedTimers,
  }
}
