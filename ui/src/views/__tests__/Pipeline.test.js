import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import Pipeline from '../Pipeline.vue'

const { mockApi, toastSuccess, toastError } = vi.hoisted(() => ({
  mockApi: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
}))

vi.mock('../../utils/api', () => ({
  api: mockApi,
}))

vi.mock('../../utils/toast', () => ({
  useToast: () => ({
    success: toastSuccess,
    error: toastError,
  }),
}))

const router = createRouter({
  history: createWebHistory(),
  routes: [{ path: '/pipeline', component: { template: '<div />' } }],
})

async function mountPipeline() {
  await router.push('/pipeline')
  await router.isReady()
  const wrapper = mount(Pipeline, {
    global: { plugins: [router] },
  })
  await flushPromises()
  return wrapper
}

describe('Pipeline', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('renders run button when idle', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '2025-01-01' })
    const wrapper = await mountPipeline()
    const runButton = wrapper.get('[data-testid="pipeline-run-button"]')
    expect(runButton.text()).toContain('Run update')
    expect(runButton.attributes('disabled')).toBeUndefined()
  })

  it('shows last update date from API', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '2025-06-15' })
    const wrapper = await mountPipeline()
    expect(wrapper.get('[data-testid="pipeline-last-update"]').text()).toContain('2025-06-15')
  })

  it('renders advanced settings section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = await mountPipeline()
    expect(wrapper.get('[data-testid="pipeline-advanced-settings"]').text()).toContain('Advanced settings')
    expect(wrapper.get('[data-testid="pipeline-data-dir-input"]').element.value).toBe('~/.quant_master/quant_master_data/tdx_cn_data')
  })

  it('renders data status section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = await mountPipeline()
    expect(wrapper.get('[data-testid="pipeline-status-label"]').text()).toBe('Data status')
    expect(wrapper.get('[data-testid="pipeline-status-value"]').text()).toBe('Ready')
  })

  it('renders run history section', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = await mountPipeline()
    expect(wrapper.get('[data-testid="pipeline-history-title"]').text()).toBe('Run history')
  })

  it('shows empty history message when no runs', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = await mountPipeline()
    expect(wrapper.get('[data-testid="pipeline-empty-history"]').text()).toBe('No runs yet')
  })

  it('sets running state on button click', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    mockApi.mockResolvedValueOnce({ runId: 'test-123' })
    const wrapper = await mountPipeline()
    expect(wrapper.vm.running).toBe(false)
    await wrapper.get('[data-testid="pipeline-run-button"]').trigger('click')
    await flushPromises()
    expect(wrapper.vm.running).toBe(true)
  })

  it('posts only the effective data dir config', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--', dataDir: '/tmp/data' })
    mockApi.mockResolvedValueOnce({ runId: 'test-123', dataDir: '/tmp/data' })
    const wrapper = await mountPipeline()
    await wrapper.get('[data-testid="pipeline-run-button"]').trigger('click')
    await flushPromises()
    expect(mockApi).toHaveBeenNthCalledWith(2, '/api/pipeline/run', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ data_dir: '/tmp/data' }),
    }))
  })

  it('stops and records a real error when run startup fails', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    mockApi.mockResolvedValueOnce(null)
    const wrapper = await mountPipeline()

    await wrapper.get('[data-testid="pipeline-run-button"]').trigger('click')
    await flushPromises()

    expect(wrapper.vm.running).toBe(false)
    expect(toastError).toHaveBeenCalledWith('Startup failed: backend unavailable')
    expect(mockApi).toHaveBeenCalledTimes(2)
    expect(JSON.parse(localStorage.getItem('pipeline_history'))).toEqual([
      expect.objectContaining({
        success: false,
        error: 'Startup failed: backend unavailable',
      }),
    ])
  })

  it('uses the canonical tdx_cn_data default before status hydration', () => {
    mockApi.mockResolvedValue({ lastUpdate: '--' })
    const wrapper = mount(Pipeline, {
      global: { plugins: [router] },
    })
    expect(wrapper.vm.cfg.data_dir).toBe('~/.quant_master/quant_master_data/tdx_cn_data')
  })

  it('renders compact history table headers', async () => {
    mockApi.mockResolvedValueOnce({ lastUpdate: '--' })
    const wrapper = await mountPipeline()
    const headers = wrapper.findAll('th')
    expect(headers).toHaveLength(4)
  })
})
