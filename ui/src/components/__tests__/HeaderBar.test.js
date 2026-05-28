import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import HeaderBar from '../HeaderBar.vue'

describe('HeaderBar', () => {
  it('renders title prop', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: '测试页面', isDark: false },
    })
    expect(wrapper.text()).toContain('测试页面')
  })

  it('displays version badge', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    expect(wrapper.text()).toContain('v2.1.0')
  })

  it('emits toggle-dark on dark mode button click', async () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    const toggleBtn = wrapper.find('button[aria-label="切换到深色模式"]')
    expect(toggleBtn.exists()).toBe(true)
    await toggleBtn.trigger('click')
    expect(wrapper.emitted('toggle-dark')).toBeTruthy()
  })

  it('shows sun icon when isDark is true', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: true },
    })
    const toggleBtn = wrapper.find('button[aria-label="切换到浅色模式"]')
    expect(toggleBtn.exists()).toBe(true)
    // Sun icon should be present (w-5 h-5 text-amber-400)
    const sunIcon = toggleBtn.find('svg.text-amber-400')
    expect(sunIcon.exists()).toBe(true)
  })

  it('shows moon icon when isDark is false', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    const toggleBtn = wrapper.find('button[aria-label="切换到深色模式"]')
    const moonIcon = toggleBtn.find('svg.text-slate-500')
    expect(moonIcon.exists()).toBe(true)
  })

  it('has mobile menu button', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    const menuBtn = wrapper.find('button[aria-label="菜单"]')
    expect(menuBtn.exists()).toBe(true)
  })

  it('emits toggle-mobile on menu button click', async () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    const menuBtn = wrapper.find('button[aria-label="菜单"]')
    await menuBtn.trigger('click')
    expect(wrapper.emitted('toggle-mobile')).toBeTruthy()
  })

  it('renders avatar with Q letter', () => {
    const wrapper = mount(HeaderBar, {
      props: { title: 'Test', isDark: false },
    })
    expect(wrapper.text()).toContain('Q')
  })
})
