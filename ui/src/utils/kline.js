/**
 * Shared K-line / market data utilities used across views.
 */

export const MINUTE_PERIODS = ['1min']
export const REALTIME_DAY_REFRESH_MS = 5000

export function isMinutePeriod(value) {
  return MINUTE_PERIODS.includes(value)
}

export function isDailyLikePeriod(value) {
  return ['D', 'W', 'M'].includes(value)
}

/**
 * Aggregate daily kline data into weekly or monthly bars.
 * @param {Array} data - array of { date, open, high, low, close, volume }
 * @param {string} mode - 'W' for weekly, 'M' for monthly
 */
export function aggregateKline(data, mode) {
  if (!data.length) return []
  const groups = {}
  const getKey = mode === 'W'
    ? (item) => {
        const date = new Date(item.date)
        const day = date.getDay()
        const monday = new Date(date)
        monday.setDate(date.getDate() - ((day + 6) % 7))
        return monday.toISOString().slice(0, 10)
      }
    : (item) => item.date.slice(0, 7)

  data.forEach((item) => {
    const key = getKey(item)
    if (!groups[key]) groups[key] = []
    groups[key].push(item)
  })

  return Object.keys(groups).sort().map((key) => {
    const group = groups[key]
    return {
      date: group[group.length - 1].date,
      open: group[0].open,
      high: Math.max(...group.map((item) => item.high)),
      low: Math.min(...group.map((item) => item.low)),
      close: group[group.length - 1].close,
      volume: group.reduce((sum, item) => sum + item.volume, 0),
    }
  })
}
