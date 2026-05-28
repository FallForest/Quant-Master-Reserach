import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useDarkMode } from '../darkMode.js'

describe('useDarkMode', () => {
  let dm

  beforeEach(() => {
    dm = useDarkMode()
    // Reset module-level isDark
    dm.isDark.value = false
    // Clear document classes
    document.documentElement.classList.remove('dark')
    document.body.classList.remove('dark')
    // Clear localStorage
    localStorage.clear()
    // Default matchMedia mock
    window.matchMedia = vi.fn().mockReturnValue({ matches: false })
  })

  it('initDarkMode reads "true" from localStorage -> isDark true', () => {
    localStorage.setItem('qm-dark-mode', 'true')
    dm.initDarkMode()
    expect(dm.isDark.value).toBe(true)
  })

  it('initDarkMode reads "false" from localStorage -> isDark false', () => {
    localStorage.setItem('qm-dark-mode', 'false')
    dm.initDarkMode()
    expect(dm.isDark.value).toBe(false)
  })

  it('initDarkMode no localStorage, matchMedia false -> isDark false', () => {
    dm.initDarkMode()
    expect(dm.isDark.value).toBe(false)
  })

  it('toggleDarkMode flips isDark', () => {
    expect(dm.isDark.value).toBe(false)
    dm.toggleDarkMode()
    expect(dm.isDark.value).toBe(true)
    dm.toggleDarkMode()
    expect(dm.isDark.value).toBe(false)
  })

  it('toggleDarkMode persists to localStorage', () => {
    dm.toggleDarkMode()
    expect(localStorage.getItem('qm-dark-mode')).toBe('true')
    dm.toggleDarkMode()
    expect(localStorage.getItem('qm-dark-mode')).toBe('false')
  })

  it('toggleDarkMode applies "dark" class to documentElement', () => {
    dm.toggleDarkMode()
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    dm.toggleDarkMode()
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })
})
