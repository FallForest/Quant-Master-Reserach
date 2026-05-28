import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import CandlestickChart from '../CandlestickChart.js'

const origCreateElement = document.createElement.bind(document)

function makeCanvas() {
  const canvas = origCreateElement('canvas')
  canvas.width = 800
  canvas.height = 400
  const ctx = {
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    setLineDash: vi.fn(),
    strokeRect: vi.fn(),
    fill: vi.fn(),
    setTransform: vi.fn(),
    createLinearGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
    measureText: vi.fn().mockReturnValue({ width: 50 }),
  }
  canvas.getContext = vi.fn().mockReturnValue(ctx)
  return canvas
}

function createWrapper() {
  const wrapper = origCreateElement('div')
  Object.defineProperty(wrapper, 'clientWidth', { value: 800 })
  Object.defineProperty(wrapper, 'clientHeight', { value: 400 })
  return wrapper
}

describe('CandlestickChart', () => {
  let chart

  beforeEach(() => {
    vi.spyOn(document, 'createElement').mockImplementation((tag) => {
      if (tag === 'canvas') return makeCanvas()
      return origCreateElement(tag)
    })

    const wrapper = createWrapper()
    chart = new CandlestickChart(wrapper)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('constructor creates canvas elements in wrapper', () => {
    expect(chart.wrapper.children.length).toBe(4)
    expect(chart.canvas).toBeTruthy()
    expect(chart.volCanvas).toBeTruthy()
    expect(chart.crosshairLabel).toBeTruthy()
    expect(chart.ohlcvLabel).toBeTruthy()
  })

  it('setData with 25 points computes ma5 at index 24 as number and null at index 3', () => {
    const data = Array.from({ length: 25 }, (_, i) => ({
      date: `2024-01-${String(i + 1).padStart(2, '0')}`,
      open: 10 + i,
      high: 11 + i,
      low: 9 + i,
      close: 10 + i,
      volume: 1000 + i * 100,
    }))
    chart.setData(data)
    expect(chart.data[24].ma5).toBeTypeOf('number')
    expect(chart.data[3].ma5).toBeNull()
  })

  it('_parseMinuteTime("2024-01-01 10:30") returns 630', () => {
    expect(chart._parseMinuteTime('2024-01-01 10:30')).toBe(630)
  })

  it('_parseMinuteTime("2024-01-01 09:30") returns 570', () => {
    expect(chart._parseMinuteTime('2024-01-01 09:30')).toBe(570)
  })

  it('_minuteRatio(570) returns 0.0 (start of session)', () => {
    expect(chart._minuteRatio(570)).toBe(0)
  })

  it('_minuteRatio(690) returns 0.5 (end of morning = 120min / 240min)', () => {
    // 690 = 11:30, end of morning session
    // elapsed = 0, minuteOfDay - start = 690 - 570 = 120
    // ratio = 120 / 240 = 0.5
    expect(chart._minuteRatio(690)).toBe(0.5)
  })

  it('_minuteRatio(810) returns (120+30)/240 (afternoon session)', () => {
    // 810 = 13:30, afternoon session starts at 780 (13:00)
    // elapsed = 120 (morning), offset = 810 - 780 = 30
    // ratio = (120 + 30) / 240 = 150/240
    expect(chart._minuteRatio(810)).toBe(150 / 240)
  })

  it('_aggregateMinute with 6 data points and interval=3 returns 2 results', () => {
    const data = [
      { date: '2024-01-01 09:30', open: 10, high: 12, low: 9, close: 11, volume: 100 },
      { date: '2024-01-01 09:31', open: 11, high: 13, low: 10, close: 12, volume: 200 },
      { date: '2024-01-01 09:32', open: 12, high: 14, low: 11, close: 13, volume: 300 },
      { date: '2024-01-01 09:33', open: 13, high: 15, low: 12, close: 14, volume: 400 },
      { date: '2024-01-01 09:34', open: 14, high: 16, low: 13, close: 15, volume: 500 },
      { date: '2024-01-01 09:35', open: 15, high: 17, low: 14, close: 16, volume: 600 },
    ]
    const result = chart._aggregateMinute(data, 3)
    expect(result).toHaveLength(2)
    expect(result[0].open).toBe(10)
    expect(result[0].high).toBe(14)
    expect(result[0].low).toBe(9)
    expect(result[0].close).toBe(13)
    expect(result[0].volume).toBe(600)
    expect(result[1].open).toBe(13)
    expect(result[1].high).toBe(17)
    expect(result[1].low).toBe(12)
    expect(result[1].close).toBe(16)
    expect(result[1].volume).toBe(1500)
  })

  it('scrollToLatest sets offset to show latest data', () => {
    const data = Array.from({ length: 200 }, (_, i) => ({
      date: `2024-01-01`,
      open: 10,
      high: 11,
      low: 9,
      close: 10,
      volume: 100,
    }))
    chart.setData(data)
    chart.scrollToLatest()
    expect(chart.offset).toBe(200 - chart.visibleCount)
  })

  it('destroy clears wrapper innerHTML', () => {
    chart.destroy()
    expect(chart.wrapper.innerHTML).toBe('')
  })
})
