import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import Overview from '../Overview.vue'

// Mock echarts
vi.mock('echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    dispose: vi.fn(),
  })),
  graphic: {
    LinearGradient: vi.fn(() => ({})),
  },
}))

// Must use vi.hoisted to create mock fns accessible by vi.mock
const { mockApi } = vi.hoisted(() => ({
  mockApi: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
  fmtNum: (n) => String(n),
}))

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: { template: '<div />' } },
    { path: '/pipeline', component: { template: '<div />' } },
    { path: '/browser', component: { template: '<div />' } },
    { path: '/factor', component: { template: '<div />' } },
    { path: '/model-lab', component: { template: '<div />' } },
    { path: '/stock-select', component: { template: '<div />' } },
    { path: '/backtest', component: { template: '<div />' } },
    { path: '/experiments', component: { template: '<div />' } },
  ],
})

describe('Overview', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state initially', () => {
    mockApi.mockResolvedValue(null)
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    expect(wrapper.vm.loading).toBe(true)
  })

  it('renders stats after API response', async () => {
    mockApi.mockResolvedValue({
      stockCount: 5200,
      calendarDays: 4500,
      lastUpdate: '2025-06-15',
      completeness: 99.8,
      fieldStats: [],
    })
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('5200')
    expect(wrapper.text()).toContain('4500')
    expect(wrapper.text()).toContain('2025-06-15')
  })

  it('renders quick action buttons', async () => {
    mockApi.mockResolvedValue(null)
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    const text = wrapper.text()
    expect(text).toContain('浏览数据')
    expect(text).toContain('因子分析')
    expect(text).toContain('模型工坊')
    expect(text).toContain('策略回测')
  })

  it('renders 6 quick action buttons', async () => {
    mockApi.mockResolvedValue(null)
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    const actionButtons = wrapper.findAll('button[aria-label]')
    expect(actionButtons.length).toBe(6)
  })

  it('renders stat cards with correct labels', async () => {
    mockApi.mockResolvedValue({
      stockCount: 100,
      calendarDays: 250,
      lastUpdate: '2025-01-01',
      completeness: 99.5,
    })
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    const text = wrapper.text()
    expect(text).toContain('股票总数')
    expect(text).toContain('交易日历')
    expect(text).toContain('最后更新')
    expect(text).toContain('数据完整度')
  })

  it('renders completeness chart section', async () => {
    mockApi.mockResolvedValue(null)
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('数据完整度')
  })

  it('sets loading to false after API call', async () => {
    mockApi.mockResolvedValue({
      stockCount: 100,
      calendarDays: 250,
    })
    const wrapper = mount(Overview, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.vm.loading).toBe(false)
  })
})
