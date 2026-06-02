import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import Position from '../Position.vue'

vi.mock('echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    dispose: vi.fn(),
    resize: vi.fn(),
  })),
}))

const { mockApi } = vi.hoisted(() => ({
  mockApi: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
}))

const positionPayload = {
  date: '2025-02-20',
  cash: 50000,
  totalAssets: 120000,
  totalMarketValue: 70000,
  totalPnl: 2500,
  totalPnlPct: 3.7,
  positionCount: 1,
  positions: [
    {
      instrument: 'SH600001',
      name: 'Test Stock A',
      shares: 1000,
      costPrice: 10,
      currentPrice: 10.5,
      marketValue: 10500,
      pnl: 500,
      pnlPct: 5,
      weight: 8.75,
    },
  ],
}

const historyPayload = {
  orders: [
    { date: '2025-02-20', instrument: 'SH600001', side: 'buy', shares: 1000, price: 10.0, status: '已成交' },
  ],
}

const executionConfig = {
  defaultBrokerKind: 'paper',
  defaultDryRun: true,
  supportedBrokers: ['paper', 'tdx'],
  liveTradingEnabled: false,
  tradeUnit: 100,
  riskDefaults: { maxOrderValue: null, maxPositionRatio: 1 },
}

const executionHistory = {
  runs: [
    {
      historyId: 'run-1',
      brokerKind: 'paper',
      dryRun: true,
      submittedAt: '2026-06-02T12:00:00',
      summary: { accepted: 1, rejected: 0 },
    },
  ],
}

const executionPreview = {
  orders: [
    { stockId: 'SH600001', side: 'buy', price: 10, amount: 800, orderValue: 8000, valid: true, validationError: '' },
  ],
  summary: { totalOrders: 1, validOrders: 1, invalidOrders: 0, buyAmount: 8000, sellAmount: 0 },
}

const executionSubmit = {
  brokerKind: 'paper',
  dryRun: true,
  summary: { total: 1, accepted: 1, rejected: 0 },
  results: [
    { stockId: 'SH600001', side: 'buy', accepted: true, orderId: 'PAPER-000001', status: 'pending', postCheckStatus: 'order_id_received', rejectionReason: '' },
  ],
}

describe('Position execution workspace', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
  })

  function mountView() {
    return mount(Position)
  }

  it('loads positions and execution config on mount', async () => {
    mockApi
      .mockResolvedValueOnce(positionPayload)
      .mockResolvedValueOnce(historyPayload)
      .mockResolvedValueOnce(executionConfig)
      .mockResolvedValueOnce(executionHistory)

    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.get('[data-testid="execution-workspace"]').text()).toContain('执行工作台')
    expect(wrapper.get('[data-testid="execution-safety-banner"]').text()).toContain('paper / dry-run')
    expect(wrapper.find('[data-testid="execution-history-empty"]').exists()).toBe(false)
  })

  it('imports execution draft and requests preview', async () => {
    sessionStorage.setItem('executionDraft', JSON.stringify({
      source: 'buffered-rebalance',
      alias: 'demo-model',
      tradeDate: '2025-02-20',
      trades: [{ instrument: 'SH600001', side: 'buy', deltaShares: 800, currentPrice: 10 }],
      summary: { estimatedBuyAmount: 8000, estimatedSellAmount: 0, estimatedFees: 10 },
    }))

    mockApi
      .mockResolvedValueOnce(positionPayload)
      .mockResolvedValueOnce(historyPayload)
      .mockResolvedValueOnce(executionConfig)
      .mockResolvedValueOnce(executionHistory)
      .mockResolvedValueOnce(executionPreview)

    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-testid="execution-toggle"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="execution-draft-meta"]').text()).toContain('demo-model')
    expect(mockApi).toHaveBeenNthCalledWith(5, '/api/execution/preview', expect.objectContaining({
      method: 'POST',
    }))
    expect(wrapper.text()).toContain('SH600001')
    expect(wrapper.text()).toContain('可提交')
  })

  it('submits execution preview with paper dry-run defaults', async () => {
    sessionStorage.setItem('executionDraft', JSON.stringify({
      source: 'buffered-rebalance',
      alias: 'demo-model',
      tradeDate: '2025-02-20',
      trades: [{ instrument: 'SH600001', side: 'buy', deltaShares: 800, currentPrice: 10 }],
      summary: { estimatedBuyAmount: 8000, estimatedSellAmount: 0, estimatedFees: 10 },
    }))

    mockApi
      .mockResolvedValueOnce(positionPayload)
      .mockResolvedValueOnce(historyPayload)
      .mockResolvedValueOnce(executionConfig)
      .mockResolvedValueOnce(executionHistory)
      .mockResolvedValueOnce(executionPreview)
      .mockResolvedValueOnce(executionSubmit)
      .mockResolvedValueOnce(executionHistory)
      .mockResolvedValueOnce(positionPayload)
      .mockResolvedValueOnce(historyPayload)

    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-testid="execution-toggle"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="execution-submit-button"]').trigger('click')
    await flushPromises()

    expect(mockApi).toHaveBeenNthCalledWith(6, '/api/execution/submit', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        brokerKind: 'paper',
        dryRun: true,
        risk: { maxOrderValue: null, maxPositionRatio: 1 },
        confirm: true,
        orders: executionPreview.orders,
      }),
    }))
    expect(wrapper.text()).toContain('order_id_received')
  })

  it('renders execution error when preview request fails', async () => {
    sessionStorage.setItem('executionDraft', JSON.stringify({
      source: 'buffered-rebalance',
      alias: 'demo-model',
      tradeDate: '2025-02-20',
      trades: [{ instrument: 'SH600001', side: 'buy', deltaShares: 800, currentPrice: 10 }],
    }))

    mockApi
      .mockResolvedValueOnce(positionPayload)
      .mockResolvedValueOnce(historyPayload)
      .mockResolvedValueOnce(executionConfig)
      .mockResolvedValueOnce(executionHistory)
      .mockResolvedValueOnce({ error: 'preview failed' })

    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.get('[data-testid="execution-error"]').text()).toContain('preview failed')
  })
})
