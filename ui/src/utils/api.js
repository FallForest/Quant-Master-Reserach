export async function api(url, opts) {
  try {
    const res = await fetch(url, opts)
    const contentType = res.headers?.get?.('content-type') || ''
    let payload = null

    if (contentType.includes('application/json')) {
      payload = await res.json()
    } else {
      const text = await res.text()
      payload = text ? { error: text } : null
    }

    if (!res.ok) {
      if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
        return {
          ...payload,
          _httpStatus: res.status,
          _httpStatusText: res.statusText,
        }
      }
      return {
        error: `${res.status} ${res.statusText}`,
        _httpStatus: res.status,
        _httpStatusText: res.statusText,
      }
    }

    return payload
  } catch (e) {
    // AbortError 是正常的取消行为，不需要记录错误
    if (e.name !== 'AbortError') {
      console.error('API error:', e)
    }
    return { error: e.message || 'Network error', _httpStatus: 0, _httpStatusText: 'Network Error', aborted: e.name === 'AbortError' }
  }
}

export function fmtNum(n, decimals) {
  if (n == null || isNaN(n)) return '--'
  if (typeof decimals === 'number') return n.toFixed(decimals)
  if (Math.abs(n) >= 1e8) return (n / 1e8).toFixed(2) + '亿'
  if (Math.abs(n) >= 1e4) return (n / 1e4).toFixed(2) + '万'
  return n.toLocaleString('zh-CN')
}
