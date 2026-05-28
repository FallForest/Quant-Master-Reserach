import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { api, fmtNum } from '../api.js'

describe('fmtNum', () => {
  it('returns "--" for null', () => {
    expect(fmtNum(null)).toBe('--')
  })

  it('returns "--" for undefined', () => {
    expect(fmtNum(undefined)).toBe('--')
  })

  it('returns "--" for NaN', () => {
    expect(fmtNum(NaN)).toBe('--')
  })

  it('formats with fixed decimals when specified', () => {
    expect(fmtNum(3.14159, 2)).toBe('3.14')
  })

  it('formats values >= 1e8 with 亿 suffix', () => {
    expect(fmtNum(200000000)).toBe('2.00亿')
  })

  it('formats values >= 1e4 with 万 suffix', () => {
    expect(fmtNum(50000)).toBe('5.00万')
  })

  it('returns a string for small numbers', () => {
    const result = fmtNum(1234)
    expect(typeof result).toBe('string')
  })

  it('handles negative values >= 1e8', () => {
    expect(fmtNum(-200000000)).toBe('-2.00亿')
  })
})

describe('api', () => {
  let fetchMock

  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('returns parsed JSON on success', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ a: 1 }),
    })
    const result = await api('/test')
    expect(result).toEqual({ a: 1 })
  })

  it('returns null on non-ok response', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    })
    const result = await api('/test')
    expect(result).toBeNull()
  })

  it('returns null when fetch throws', async () => {
    fetchMock.mockRejectedValue(new Error('Network error'))
    const result = await api('/test')
    expect(result).toBeNull()
  })
})
