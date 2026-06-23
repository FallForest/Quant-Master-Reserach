/**
 * Shared formatting utilities used across views.
 */

const DEFAULT_LOCALE = 'zh-CN'

export function fmtAmount(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  return Number(n).toLocaleString(DEFAULT_LOCALE, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

export function fmtPct(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  const sign = n > 0 ? '+' : ''
  return sign + Number(n).toFixed(2) + '%'
}

export function fmtPrice(n) {
  if (n == null || Number.isNaN(Number(n))) return '--'
  return Number(n).toFixed(2)
}

export function pnlClass(n) {
  if (n > 0) return 'text-bull'
  if (n < 0) return 'text-bear'
  return 'text-slate-500'
}

export function executionSideLabel(side) {
  return side === 'buy' ? '买入' : side === 'sell' ? '卖出' : side
}

export function executionSideClass(side) {
  return side === 'buy'
    ? 'bg-bull/10 text-bull'
    : 'bg-bear/10 text-bear'
}

export function sideText(side) {
  return side === 'buy' ? '买入' : side === 'sell' ? '卖出' : '持有'
}

export function sideClass(side) {
  return side === 'buy'
    ? 'bg-bull/10 text-bull'
    : side === 'sell'
      ? 'bg-bear/10 text-bear'
      : 'bg-surface-2 text-slate-500'
}
