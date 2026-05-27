import { fmtNum } from '../utils/api'

// A 股交易时段（分钟数从 00:00 起）
const TRADING_SESSIONS = [
  { start: 9 * 60 + 30, end: 11 * 60 + 30 },   // 9:30-11:30
  { start: 13 * 60,      end: 15 * 60 },          // 13:00-15:00
]
const SESSION_TOTAL = TRADING_SESSIONS.reduce((s, t) => s + (t.end - t.start), 0) // 240 min

export default class CandlestickChart {
  constructor(wrapper, opts = {}) {
    this.wrapper = wrapper
    wrapper.classList.add('chart-container')
    wrapper.style.position = 'relative'

    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')
    wrapper.appendChild(this.canvas)

    this.volCanvas = document.createElement('canvas')
    this.volCtx = this.volCanvas.getContext('2d')
    wrapper.appendChild(this.volCanvas)

    this.crosshairLabel = document.createElement('div')
    this.crosshairLabel.className = 'chart-crosshair-label bg-brand-950/90 text-white hidden'
    wrapper.appendChild(this.crosshairLabel)

    this.ohlcvLabel = document.createElement('div')
    this.ohlcvLabel.className = 'absolute top-1 left-2 text-xs font-mono leading-relaxed pointer-events-none z-10'
    this.ohlcvLabel.style.color = '#CBD5E1'
    wrapper.appendChild(this.ohlcvLabel)

    const t = opts.theme || {}
    this.theme = {
      bull: t.bull || '#EF4444',
      bear: t.bear || '#10B981',
      bg:   t.bg   || '#FFFFFF',
      grid: t.grid || '#E2E8F0',
      text: t.text || '#475569',
      maColors: t.maColors || ['#F59E0B', '#3B82F6', '#A855F7'],
    }

    this.data = []
    this.visibleCount = 120
    this.offset = 0
    this.hoverIdx = -1
    this.period = 'D'

    this.PADDING = { top: 24, right: 64, bottom: 28, left: 12 }
    this.VOL_HEIGHT = 80
    this.VOL_GAP = 8

    this._onResize = this.resize.bind(this)
    this._onMouse  = this._handleMouse.bind(this)
    this._onLeave  = this._handleLeave.bind(this)
    this._onWheel  = this._handleWheel.bind(this)

    window.addEventListener('resize', this._onResize)
    this.canvas.addEventListener('mousemove', this._onMouse)
    this.canvas.addEventListener('mouseleave', this._onLeave)
    this.canvas.addEventListener('wheel', this._onWheel, { passive: false })

    this.resize()
  }

  destroy() {
    window.removeEventListener('resize', this._onResize)
    this.canvas.removeEventListener('mousemove', this._onMouse)
    this.canvas.removeEventListener('mouseleave', this._onLeave)
    this.canvas.removeEventListener('wheel', this._onWheel)
    this.wrapper.innerHTML = ''
  }

  setData(raw) {
    this.data = raw
    this.visibleCount = Math.min(this.visibleCount, this.data.length)
    this.offset = Math.max(0, this.data.length - this.visibleCount)
    this.hoverIdx = -1
    this._computeMA()
    this.draw()
  }

  setPeriod(p) { this.period = p }

  // ---- 分钟时间工具 ----

  get _isMinute() {
    return ['1min', '5min', '15min', '30min', '60min'].includes(this.period)
  }

  /** 从 "YYYY-MM-DD HH:mm" 解析当天分钟数 */
  _parseMinuteTime(dateStr) {
    if (!dateStr) return -1
    const parts = dateStr.split(' ')
    if (parts.length < 2) return -1
    const [hh, mm] = parts[1].split(':').map(Number)
    return hh * 60 + mm
  }

  /** 分钟数 → [0,1] 位置（跨所有交易时段连续映射） */
  _minuteRatio(minuteOfDay) {
    let elapsed = 0
    for (const s of TRADING_SESSIONS) {
      if (minuteOfDay >= s.start && minuteOfDay <= s.end) {
        return (elapsed + (minuteOfDay - s.start)) / SESSION_TOTAL
      }
      elapsed += s.end - s.start
    }
    return -1
  }

  /** 将 chartX 区域中的时间比率映射为像素 X */
  _timeToX(ratio, chartX, chartW) {
    return chartX + ratio * chartW
  }

  /** 1min 数据聚合为 N 分钟 */
  _aggregateMinute(data, interval) {
    if (!data.length || interval <= 1) return data
    const result = []
    for (let i = 0; i < data.length; i += interval) {
      const group = data.slice(i, Math.min(i + interval, data.length))
      result.push({
        date: group[0].date,
        open: group[0].open,
        high: Math.max(...group.map(d => d.high)),
        low: Math.min(...group.map(d => d.low)),
        close: group[group.length - 1].close,
        volume: group.reduce((s, d) => s + d.volume, 0),
      })
    }
    return result
  }

  _computeMA() {
    const d = this.data
    ;[5, 10, 20].forEach(n => {
      const key = `ma${n}`
      let sum = 0
      const q = []
      for (let i = 0; i < d.length; i++) {
        sum += d[i].close
        q.push(d[i].close)
        if (q.length > n) sum -= q.shift()
        d[i][key] = q.length === n ? sum / n : null
      }
    })
  }

  resize() {
    const w = this.wrapper.clientWidth
    const mainH = Math.max(200, this.wrapper.clientHeight - this.VOL_HEIGHT - this.VOL_GAP - 8)
    const dpr = window.devicePixelRatio || 1

    this.W = w
    this.mainH = mainH
    this.canvas.width = w * dpr
    this.canvas.height = mainH * dpr
    this.canvas.style.width = w + 'px'
    this.canvas.style.height = mainH + 'px'
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    this.volCanvas.width = w * dpr
    this.volCanvas.height = this.VOL_HEIGHT * dpr
    this.volCanvas.style.width = w + 'px'
    this.volCanvas.style.height = this.VOL_HEIGHT + 'px'
    this.volCtx.setTransform(dpr, 0, 0, dpr, 0, 0)

    this.draw()
  }

  // ---- 主绘制 ----

  draw() {
    if (!this.data.length) {
      const ctx = this.ctx
      ctx.clearRect(0, 0, this.W, this.mainH)
      ctx.fillStyle = this.theme.bg
      ctx.fillRect(0, 0, this.W, this.mainH)
      ctx.fillStyle = this.theme.text
      ctx.font = '13px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(this._isMinute ? '暂无分钟数据' : '暂无数据', this.W / 2, this.mainH / 2)
      this.volCtx.clearRect(0, 0, this.W, this.VOL_HEIGHT)
      return
    }

    const { top, right, bottom, left } = this.PADDING
    const chartW = this.W - left - right
    const chartH = this.mainH - top - bottom

    if (this._isMinute) {
      this._drawMinute(chartW, chartH, top, right, bottom, left)
    } else {
      this._drawIndex(chartW, chartH, top, right, bottom, left)
    }
  }

  // ---- 日/周/月：原有 index-based 绘制 ----

  _drawIndex(chartW, chartH, top, right, bottom, left) {
    const ctx = this.ctx
    const end = Math.min(this.offset + this.visibleCount, this.data.length)
    const visible = this.data.slice(this.offset, end)
    const count = visible.length
    if (!count) return

    const gap = chartW / count
    const candleW = Math.max(1, gap * 0.65)

    const { minP, maxP, maxV } = this._priceRange(visible)
    const yScale = (p) => top + chartH * (1 - (p - minP) / (maxP - minP))
    const xScale = (i) => left + i * gap + gap / 2

    ctx.clearRect(0, 0, this.W, this.mainH)
    ctx.fillStyle = this.theme.bg
    ctx.fillRect(0, 0, this.W, this.mainH)

    this._drawPriceGrid(ctx, yScale, left, right, top, chartH, minP, maxP)

    // 日期横轴
    ctx.textAlign = 'center'
    ctx.fillStyle = this.theme.text
    ctx.font = '11px "Fira Code", monospace'
    const step = Math.max(1, Math.floor(count / 6))
    for (let i = 0; i < count; i += step) {
      const d = visible[i]
      if (d?.date) ctx.fillText(d.date.length > 5 ? d.date.slice(5) : d.date, xScale(i), this.mainH - 2)
    }

    this._drawMALines(ctx, visible, xScale, yScale)
    this._drawCandles(ctx, visible, xScale, yScale, candleW)
    this._drawVolume(visible, (i) => xScale(i), candleW, maxV)

    if (this.hoverIdx >= 0 && this.hoverIdx < count) {
      this._drawCrosshair(visible, this.hoverIdx, xScale, yScale)
    }
  }

  // ---- 分钟线：time-based 绘制 ----

  _drawMinute(chartW, chartH, top, right, bottom, left) {
    const ctx = this.ctx
    const visible = this.data  // 当天所有数据，不分页
    const count = visible.length
    if (!count) return

    const minuteTimes = visible.map(d => this._parseMinuteTime(d.date))
    const timeToX = (ratio) => this._timeToX(ratio, left, chartW)
    const minuteToX = (min) => {
      const r = this._minuteRatio(min)
      return r >= 0 ? timeToX(r) : -1
    }

    const { minP, maxP, maxV } = this._priceRange(visible)
    const yScale = (p) => top + chartH * (1 - (p - minP) / (maxP - minP))

    ctx.clearRect(0, 0, this.W, this.mainH)
    ctx.fillStyle = this.theme.bg
    ctx.fillRect(0, 0, this.W, this.mainH)

    this._drawPriceGrid(ctx, yScale, left, right, top, chartH, minP, maxP)
    this._drawTimeGrid(ctx, minuteToX, top, chartH)

    // MA 线
    ;[5, 10, 20].forEach((n, mi) => {
      ctx.strokeStyle = this.theme.maColors[mi]
      ctx.lineWidth = 1
      ctx.beginPath()
      let started = false
      visible.forEach((d, i) => {
        const v = d[`ma${n}`]
        if (v == null) return
        const x = minuteToX(minuteTimes[i])
        if (x < 0) return
        const y = yScale(v)
        if (!started) { ctx.moveTo(x, y); started = true } else ctx.lineTo(x, y)
      })
      ctx.stroke()
    })

    // 分时线 + 渐变填充
    const lineColor = '#3B82F6'
    const points = []
    visible.forEach((d, i) => {
      const x = minuteToX(minuteTimes[i])
      if (x < 0) return
      points.push({ x, y: yScale(d.close) })
    })

    if (points.length >= 2) {
      // 填充区域
      const grad = ctx.createLinearGradient(0, top, 0, top + chartH)
      grad.addColorStop(0, 'rgba(59,130,246,0.15)')
      grad.addColorStop(1, 'rgba(59,130,246,0)')
      ctx.beginPath()
      ctx.moveTo(points[0].x, top + chartH)
      points.forEach(p => ctx.lineTo(p.x, p.y))
      ctx.lineTo(points[points.length - 1].x, top + chartH)
      ctx.closePath()
      ctx.fillStyle = grad
      ctx.fill()

      // 折线
      ctx.beginPath()
      ctx.moveTo(points[0].x, points[0].y)
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y)
      }
      ctx.strokeStyle = lineColor
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    // 成交量柱
    const barW = Math.max(2, (chartW / SESSION_TOTAL) * 0.6)
    this._drawVolume(visible, (i) => minuteToX(minuteTimes[i]), barW, maxV)

    // 十字线
    if (this.hoverIdx >= 0 && this.hoverIdx < count) {
      this._drawCrosshair(visible, this.hoverIdx, (i) => minuteToX(minuteTimes[i]), yScale)
    }
  }

  // ---- 子绘制方法 ----

  _priceRange(visible) {
    let minP = Infinity, maxP = -Infinity, maxV = 0
    visible.forEach(d => {
      if (d.low < minP) minP = d.low
      if (d.high > maxP) maxP = d.high
      if (d.volume > maxV) maxV = d.volume
      ;[5, 10, 20].forEach(n => {
        const ma = d[`ma${n}`]
        if (ma != null) { if (ma < minP) minP = ma; if (ma > maxP) maxP = ma }
      })
    })
    const range = maxP - minP || 1
    return {
      minP: minP - range * 0.05,
      maxP: maxP + range * 0.05,
      maxV,
    }
  }

  _drawPriceGrid(ctx, yScale, left, right, top, chartH, minP, maxP) {
    ctx.strokeStyle = this.theme.grid
    ctx.lineWidth = 0.5
    ctx.fillStyle = this.theme.text
    ctx.font = '11px "Fira Code", monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const p = minP + (maxP - minP) * (i / 5)
      const y = Math.round(yScale(p)) + 0.5
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(this.W - right, y); ctx.stroke()
      ctx.fillText(p.toFixed(2), this.W - 4, y + 3)
    }
  }

  /** 分钟线专用：基于交易时段的竖向网格 + 时间标签 */
  _drawTimeGrid(ctx, minuteToX, top, chartH) {
    ctx.strokeStyle = this.theme.grid
    ctx.lineWidth = 0.5
    ctx.fillStyle = this.theme.text
    ctx.font = '11px "Fira Code", monospace'
    ctx.textAlign = 'center'

    const lastLabelX = -Infinity

    // 生成标签：每 30 分钟一个，加 :15/:45 辅助线（15min 步长时）
    const labels = []
    for (const session of TRADING_SESSIONS) {
      let m = session.start
      // 对齐到下一个整点或半点
      const rem = m % 30
      if (rem !== 0) m += (30 - rem)
      for (; m <= session.end; m += 30) {
        if (m === session.end && session !== TRADING_SESSIONS[TRADING_SESSIONS.length - 1]) continue
        labels.push(m)
      }
    }

    // 竖线 + 标签
    let prevLabelX = -Infinity
    for (const m of labels) {
      const x = Math.round(minuteToX(m)) + 0.5
      if (x < 0) continue
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + chartH); ctx.stroke()

      const h = Math.floor(m / 60)
      const mm = m % 60
      const label = `${String(h).padStart(2, '0')}:${String(mm).padStart(2, '0')}`
      if (x - prevLabelX >= 40) {
        ctx.fillText(label, x, this.mainH - 2)
        prevLabelX = x
      }
    }

    // 15 分钟辅助虚线
    ctx.strokeStyle = this.theme.grid
    ctx.lineWidth = 0.3
    ctx.setLineDash([2, 4])
    for (const session of TRADING_SESSIONS) {
      let m = session.start
      const rem = m % 15
      if (rem !== 0) m += (15 - rem)
      for (; m <= session.end; m += 15) {
        if (labels.includes(m)) continue  // 跳过已有实线的位置
        const x = Math.round(minuteToX(m)) + 0.5
        if (x < 0) continue
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + chartH); ctx.stroke()
      }
    }
    ctx.setLineDash([])
  }

  _drawMALines(ctx, visible, xScale, yScale) {
    ;[5, 10, 20].forEach((n, mi) => {
      ctx.strokeStyle = this.theme.maColors[mi]
      ctx.lineWidth = 1
      ctx.beginPath()
      let started = false
      visible.forEach((d, i) => {
        const v = d[`ma${n}`]
        if (v == null) return
        const x = xScale(i), y = yScale(v)
        if (!started) { ctx.moveTo(x, y); started = true } else ctx.lineTo(x, y)
      })
      ctx.stroke()
    })
  }

  _drawCandles(ctx, visible, xScale, yScale, candleW) {
    visible.forEach((d, i) => {
      const x = xScale(i)
      const bullish = d.close >= d.open
      const color = bullish ? this.theme.bull : this.theme.bear

      ctx.strokeStyle = color
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, yScale(d.high))
      ctx.lineTo(x, yScale(d.low))
      ctx.stroke()

      const yOpen = yScale(d.open)
      const yClose = yScale(d.close)
      const bodyTop = Math.min(yOpen, yClose)
      const bodyH = Math.max(1, Math.abs(yClose - yOpen))

      if (bullish) {
        ctx.fillStyle = '#FFF'
        ctx.strokeStyle = color
        ctx.lineWidth = 1
        ctx.strokeRect(x - candleW / 2, bodyTop, candleW, bodyH)
        ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH)
      } else {
        ctx.fillStyle = color
        ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH)
      }
    })
  }

  _drawVolume(visible, xAt, candleW, maxV) {
    const ctx = this.volCtx
    const h = this.VOL_HEIGHT
    ctx.clearRect(0, 0, this.W, h)
    ctx.fillStyle = this.theme.bg
    ctx.fillRect(0, 0, this.W, h)

    ctx.strokeStyle = this.theme.grid
    ctx.lineWidth = 0.5
    ctx.setLineDash([4, 4])
    const midY = Math.round(h / 2) + 0.5
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(this.W, midY); ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = this.theme.text
    ctx.font = '10px "Fira Code", monospace'
    ctx.textAlign = 'right'
    ctx.fillText(fmtNum(maxV), this.W - 4, 12)

    const vScale = (h - 4) / (maxV || 1)
    visible.forEach((d, i) => {
      const x = xAt(i)
      if (x < 0) return
      const bullish = d.close >= d.open
      const color = bullish ? this.theme.bull : this.theme.bear
      const barH = Math.max(1, d.volume * vScale)
      ctx.fillStyle = color
      ctx.globalAlpha = 0.6
      ctx.fillRect(x - candleW / 2, h - barH, candleW, barH)
      ctx.globalAlpha = 1
    })
  }

  _drawCrosshair(visible, idx, xAt, yScale) {
    const ctx = this.ctx
    const d = visible[idx]
    const x = xAt(idx)
    if (x < 0) return

    ctx.strokeStyle = '#94A3B8'
    ctx.lineWidth = 0.5
    ctx.setLineDash([4, 4])
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.mainH); ctx.stroke()

    const y = yScale(d.close)
    ctx.beginPath(); ctx.moveTo(this.PADDING.left, y); ctx.lineTo(this.W - this.PADDING.right, y); ctx.stroke()
    ctx.setLineDash([])

    const priceW = 56, priceH = 20
    ctx.fillStyle = '#1E40AF'
    ctx.fillRect(this.W - this.PADDING.right, y - priceH / 2, priceW, priceH)
    ctx.fillStyle = '#FFF'
    ctx.font = '11px "Fira Code", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(d.close.toFixed(2), this.W - this.PADDING.right + priceW / 2, y + 4)

    const chg = d.close - d.open
    const chgPct = d.open ? (chg / d.open * 100).toFixed(2) : '--'
    const color = chg >= 0 ? this.theme.bull : this.theme.bear
    this.ohlcvLabel.innerHTML =
      `<span class="font-semibold">${d.date || ''}</span>  ` +
      `<span style="color:${color}">` +
      `O ${d.open.toFixed(2)}  H ${d.high.toFixed(2)}  L ${d.low.toFixed(2)}  C ${d.close.toFixed(2)}  ` +
      `${chg >= 0 ? '+' : ''}${chg.toFixed(2)} (${chg >= 0 ? '+' : ''}${chgPct}%)</span>  ` +
      `V ${fmtNum(d.volume)}`

    this.crosshairLabel.textContent = this._isMinute
      ? (d.date?.split(' ')[1] || d.date || '')
      : (d.date || '')
    this.crosshairLabel.classList.remove('hidden')
    this.crosshairLabel.style.left = Math.min(x - 30, this.W - 80) + 'px'
    this.crosshairLabel.style.bottom = (this.VOL_HEIGHT + this.VOL_GAP + 4) + 'px'
    this.crosshairLabel.style.background = '#1E40AF'
    this.crosshairLabel.style.color = '#FFF'
  }

  // ---- 交互 ----

  _handleMouse(e) {
    const rect = this.canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const { left, right } = this.PADDING
    const chartW = this.W - left - right

    let idx
    if (this._isMinute) {
      // 时间轴反算：鼠标位置 → 时间比率 → 最近的 bar
      const ratio = (mx - left) / chartW
      const minuteOfDay = ratio * SESSION_TOTAL
      // 映射回实际分钟数
      let elapsed = 0
      let targetMinute = -1
      for (const s of TRADING_SESSIONS) {
        const len = s.end - s.start
        if (minuteOfDay < elapsed + len) {
          targetMinute = s.start + (minuteOfDay - elapsed)
          break
        }
        elapsed += len
      }
      if (targetMinute < 0) targetMinute = TRADING_SESSIONS[TRADING_SESSIONS.length - 1].end

      // 找最近的 bar
      let best = 0, bestDist = Infinity
      for (let i = 0; i < this.data.length; i++) {
        const mt = this._parseMinuteTime(this.data[i].date)
        const dist = Math.abs(mt - targetMinute)
        if (dist < bestDist) { bestDist = dist; best = i }
      }
      idx = best
    } else {
      const gap = chartW / Math.min(this.visibleCount, this.data.length - this.offset)
      idx = Math.round((mx - left - gap / 2) / gap)
      const count = Math.min(this.visibleCount, this.data.length - this.offset)
      idx = Math.max(0, Math.min(idx, count - 1))
    }

    if (idx !== this.hoverIdx) { this.hoverIdx = idx; this.draw() }
  }

  _handleLeave() {
    this.hoverIdx = -1
    this.crosshairLabel.classList.add('hidden')
    this.ohlcvLabel.innerHTML = ''
    this.draw()
  }

  _handleWheel(e) {
    e.preventDefault()
    // 分钟线单日数据不支持滚动
    if (this._isMinute) return

    const delta = e.deltaY > 0 ? -10 : 10
    if (e.ctrlKey || e.metaKey) {
      this.visibleCount = Math.max(20, Math.min(500, this.visibleCount - delta))
    } else {
      this.offset = Math.max(0, Math.min(this.data.length - this.visibleCount, this.offset + delta))
    }
    this.draw()
  }

  scrollToLatest() {
    this.offset = Math.max(0, this.data.length - this.visibleCount)
    this.draw()
  }
}
