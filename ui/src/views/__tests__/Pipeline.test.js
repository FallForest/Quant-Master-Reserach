import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import Pipeline from '../Pipeline.vue'

const { mockApi } = vi.hoisted(() => ({
  mockApi: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
}))

vi.mock('../../utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
  }),
}))

const router = createRouter({
  history: createWebHistory(),
  routes: [{ path: '/pipeline', component: { template: '<div />' } }],
})

describe('Pipeline', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('renders run button when idle', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '2025-01-01' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('一键更新')
  })

  it('shows last update date from API', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '2025-06-15' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('2025-06-15')
  })

  it('renders advanced settings section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('高级设置')
  })

  it('renders data status section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('数据状态')
    expect(wrapper.text()).toContain('数据就绪')
  })

  it('renders run history section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('运行历史')
  })

  it('shows empty history message when no runs', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('暂无运行记录')
  })

  it('sets running state on button click', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    mockApi.mockResolvedValueOnce({ runId: 'test-123' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    expect(wrapper.vm.running).toBe(false)
    const runBtn = wrapper.find('button')
    await runBtn.trigger('click')
    await flushPromises()
    expect(wrapper.vm.running).toBe(true)
  })

  it('renders source and region select options in advanced settings', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    const text = wrapper.text()
    expect(text).toContain('数据源')
    expect(text).toContain('地区')
  })

  it('renders data history table headers', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    await flushPromises()
    const headers = wrapper.findAll('th')
    expect(headers.length).toBe(5)
  })
})
