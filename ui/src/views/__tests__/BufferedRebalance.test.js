import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import BufferedRebalance from '../BufferedRebalance.vue'

const { mockApi } = vi.hoisted(() => ({
  mockApi: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
}))

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/strategy/buffered-rebalance', component: { template: '<div />' } },
    { path: '/execution', component: { template: '<div />' } },
  ],
})

const samplePayload = {
  alias: 'demo-model',
  tradeDate: '2025-02-20',
  config: { topK: 5, holdTopk: 8, rankBuffer: 3, weightMode: 'equal', riskDegree: 0.95 },
  holdings: { cash: 120000.5 },
  prediction: {
    stocks: [
      { rank: 1, instrument: 'SH600001', name: 'Test Stock A', score: 0.91 },
      { rank: 2, instrument: 'SH600002', name: 'Test Stock B', score: 0.87 },
    ],
  },
  selected: [{ instrument: 'SH600001' }],
  targetPositions: [
    { instrument: 'SH600001', bufferKept: true, isNew: false, currentWeight: 10.0, targetWeight: 19.0, score: 0.91 },
    { instrument: 'SH600002', bufferKept: false, isNew: true, currentWeight: 0.0, targetWeight: 19.0, score: 0.87 },
  ],
  trades: [
    { instrument: 'SH600001', name: 'Test Stock A', side: 'buy', currentShares: 1000, targetShares: 1800, currentWeight: 10.0, targetWeight: 19.0, tradeAmount: 8000 },
    { instrument: 'SH600003', name: 'Test Stock C', side: 'sell', currentShares: 900, targetShares: 0, currentWeight: 8.0, targetWeight: 0.0, tradeAmount: 7200 },
  ],
  summary: {
    keptCount: 1,
    newCount: 1,
    estimatedBuyAmount: 8000,
    estimatedSellAmount: 7200,
    estimatedFees: 18.6,
    turnoverPct: 12.4,
    cashAfterTrades: 119181.9,
  },
  explanation: {
    title: 'BufferedWeightStrategy 调仓预览',
    why: '保留 buffer 范围内旧仓位，降低换手。',
    how: ['读取当前持仓', '生成候选排名', '保留旧仓', '补足 topk'],
  },
}

describe('BufferedRebalance', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
  })

  async function mountView() {
    await router.push('/strategy/buffered-rebalance')
    await router.isReady()
    const wrapper = mount(BufferedRebalance, {
      global: { plugins: [router] },
    })
    await flushPromises()
    return wrapper
  }

  it('loads buffered rebalance preview and renders trades', async () => {
    mockApi.mockResolvedValue(samplePayload)
    const wrapper = await mountView()

    expect(mockApi).toHaveBeenCalled()
    expect(wrapper.text()).toContain('Buffered 调仓')
    expect(wrapper.text()).toContain('调仓建议')
    expect(wrapper.text()).toContain('SH600001')
    expect(wrapper.text()).toContain('买入')
    expect(wrapper.text()).toContain('预计费用')
  })

  it('shows strategy explanation and selected hits', async () => {
    mockApi.mockResolvedValue(samplePayload)
    const wrapper = await mountView()

    expect(wrapper.text()).toContain('策略解释')
    expect(wrapper.text()).toContain('模型选股命中')
    expect(wrapper.text()).toContain('进入目标仓')
  })

  it('stores execution draft and navigates to execution', async () => {
    mockApi.mockResolvedValue(samplePayload)
    const wrapper = await mountView()

    await wrapper.get('[data-testid="buffered-to-execution"]').trigger('click')
    await flushPromises()

    expect(router.currentRoute.value.path).toBe('/execution')
    const draft = JSON.parse(sessionStorage.getItem('executionDraft'))
    expect(draft.alias).toBe('demo-model')
    expect(draft.tradeDate).toBe('2025-02-20')
    expect(draft.trades).toHaveLength(2)
  })

  it('renders error state when api fails', async () => {
    mockApi.mockResolvedValue({ error: 'preview failed' })
    const wrapper = await mountView()

    expect(wrapper.text()).toContain('preview failed')
  })
})
