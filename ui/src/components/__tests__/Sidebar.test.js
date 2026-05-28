import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import Sidebar from '../Sidebar.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: { template: '<div />' } },
    { path: '/pipeline', component: { template: '<div />' } },
    { path: '/browser', component: { template: '<div />' } },
    { path: '/factor', component: { template: '<div />' } },
    { path: '/model-lab', component: { template: '<div />' } },
    { path: '/model-performance', component: { template: '<div />' } },
    { path: '/stock-select', component: { template: '<div />' } },
    { path: '/experiments', component: { template: '<div />' } },
    { path: '/strategy-lab', component: { template: '<div />' } },
    { path: '/backtest', component: { template: '<div />' } },
    { path: '/portfolio', component: { template: '<div />' } },
    { path: '/optimizer', component: { template: '<div />' } },
    { path: '/attribution', component: { template: '<div />' } },
  ],
})

describe('Sidebar', () => {
  it('renders all 13 navigation links', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })
    // 13 nav item buttons with aria-label
    const navButtons = wrapper.findAll('button[aria-label]')
    // 13 nav items + 1 collapse button = 14 buttons with aria-label
    expect(navButtons.length).toBe(14)
  })

  it('emits toggle event on collapse button click', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })
    const collapseBtn = wrapper.find('button[aria-label="收起侧边栏"]')
    expect(collapseBtn.exists()).toBe(true)
    await collapseBtn.trigger('click')
    expect(wrapper.emitted('toggle')).toBeTruthy()
  })

  it('applies sidebar-collapsed class when collapsed prop is true', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: true, mobileOpen: false },
      global: { plugins: [router] },
    })
    const aside = wrapper.find('aside')
    expect(aside.classes()).toContain('sidebar-collapsed')
  })

  it('applies sidebar-expanded class when not collapsed', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })
    const aside = wrapper.find('aside')
    expect(aside.classes()).toContain('sidebar-expanded')
  })

  it('renders all 5 group labels', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })
    const text = wrapper.text()
    expect(text).toContain('概览')
    expect(text).toContain('数据')
    expect(text).toContain('模型')
    expect(text).toContain('策略')
    expect(text).toContain('分析')
  })

  it('renders QuantMaster logo text', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })
    expect(wrapper.text()).toContain('QuantMaster')
  })
})
