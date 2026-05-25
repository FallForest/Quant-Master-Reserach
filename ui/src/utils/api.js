export async function api(url, opts) {
  try {
    const res = await fetch(url, opts)
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return await res.json()
  } catch (e) {
    console.error('API error:', e)
    return null
  }
}

export function fmtNum(n, decimals) {
  if (n == null || isNaN(n)) return '--'
  if (typeof decimals === 'number') return n.toFixed(decimals)
  if (Math.abs(n) >= 1e8) return (n / 1e8).toFixed(2) + '亿'
  if (Math.abs(n) >= 1e4) return (n / 1e4).toFixed(2) + '万'
  return n.toLocaleString('zh-CN')
}
