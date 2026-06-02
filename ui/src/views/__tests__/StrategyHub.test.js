import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import StrategyHub from '../StrategyHub.vue'

const { mockApi } = vi.hoisted(() => ({
  mockApi: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
}))

const samplePayload = {
  alias: 'demo-model',
  tradeDate: '2025-02-20',
  config: { topK: 5, holdTopk: 8, rankBuffer: 3, weightMode: 'equal', riskDegree: 0.95 },
  holdings: { cash: 120000.5 },
  prediction: {
    stocks: [
      { rank: 1, instrument: 'SH600001', name: 'Test Stock A', score: 0.91 },
    ],
  },
  selected: [{ instrument: 'SH600001' }],
  targetPositions: [
    { instrument: 'SH600001', bufferKept: true, isNew: false, currentWeight: 10.0, targetWeight: 19.0, score: 0.91 },
  ],
  trades: [
    { instrument: 'SH600001', name: 'Test Stock A', side: 'buy', currentShares: 1000, targetShares: 1800, currentWeight: 10.0, targetWeight: 19.0, tradeAmount: 8000 },
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
    how: ['读取当前持仓', '生成候选排名'],
  },
}

function makeRouter() {
  return createRouter({
    history: createWebHistory(),
    routes: [
      { path: '/strategy', component: StrategyHub },
      { path: '/strategy/buffered-rebalance', component: StrategyHub },
      { path: '/strategy-buffered-rebalance', redirect: '/strategy/buffered-rebalance' },
    ],
  })
}

describe('StrategyHub', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.mockResolvedValue(samplePayload)
  })

  it('redirects /strategy to the default buffered strategy route', async () => {
    const router = makeRouter()
    router.push('/strategy')
    await router.isReady()

    mount(StrategyHub, {
      global: { plugins: [router] },
    })

    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/strategy/buffered-rebalance')
  })

  it('redirects legacy buffered route to the canonical strategy route', async () => {
    const router = makeRouter()
    router.push('/strategy-buffered-rebalance')
    await router.isReady()

    mount(StrategyHub, {
      global: { plugins: [router] },
    })

    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/strategy/buffered-rebalance')
  })

  it('renders the strategy hub header and buffered strategy content', async () => {
    const router = makeRouter()
    router.push('/strategy/buffered-rebalance')
    await router.isReady()

    const wrapper = mount(StrategyHub, {
      global: { plugins: [router] },
    })

    await flushPromises()
    expect(wrapper.text()).toContain('策略调仓')
    expect(wrapper.text()).toContain('Buffered 调仓')
    expect(wrapper.text()).toContain('当前已接入 1 个策略')
    expect(wrapper.text()).toContain('调仓建议')
  })

  it('keeps buffered strategy selected when deep linked', async () => {
    const router = makeRouter()
    router.push('/strategy/buffered-rebalance')
    await router.isReady()

    const wrapper = mount(StrategyHub, {
      global: { plugins: [router] },
    })

    await flushPromises()
    const button = wrapper.find('button[aria-label="Buffered 调仓"]')
    expect(button.attributes('aria-current')).toBe('page')
  })
})
