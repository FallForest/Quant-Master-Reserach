import { describe, it, expect } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import Sidebar from '../Sidebar.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: { template: '<div />' } },
    { path: '/browser', component: { template: '<div />' } },
    { path: '/model', component: { template: '<div />' } },
    { path: '/strategy', component: { template: '<div />' } },
    { path: '/strategy/buffered-rebalance', component: { template: '<div />' } },
    { path: '/execution', component: { template: '<div />' } },
  ],
})

describe('Sidebar', () => {
  it('renders all 5 navigation links', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })

    const navButtons = wrapper.findAll('nav button[aria-label]')
    expect(navButtons.length).toBe(5)
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

    expect(wrapper.find('aside').classes()).toContain('sidebar-collapsed')
  })

  it('applies sidebar-expanded class when not collapsed', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })

    expect(wrapper.find('aside').classes()).toContain('sidebar-expanded')
  })

  it('renders all 4 group labels', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })

    const text = wrapper.text()
    expect(text).toContain('总览')
    expect(text).toContain('数据')
    expect(text).toContain('策略')
    expect(text).toContain('执行')
  })

  it('keeps 策略调仓 active for nested strategy routes', async () => {
    router.push('/strategy/buffered-rebalance')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })

    await flushPromises()
    const strategyButton = wrapper.find('button[aria-label="策略调仓"]')
    expect(strategyButton.attributes('aria-current')).toBe('page')
  })

  it('navigates to /strategy when clicking 策略调仓', async () => {
    router.push('/')
    await router.isReady()
    const wrapper = mount(Sidebar, {
      props: { collapsed: false, mobileOpen: false },
      global: { plugins: [router] },
    })

    await wrapper.find('button[aria-label="策略调仓"]').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/strategy')
  })

})
