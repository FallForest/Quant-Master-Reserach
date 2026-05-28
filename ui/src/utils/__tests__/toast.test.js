import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useToast } from '../toast.js'

describe('useToast', () => {
  let toast

  beforeEach(() => {
    toast = useToast()
    toast.toasts.value = []
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('show("hello") adds 1 toast with msg and type info', () => {
    toast.show('hello')
    expect(toast.toasts.value).toHaveLength(1)
    expect(toast.toasts.value[0].msg).toBe('hello')
    expect(toast.toasts.value[0].type).toBe('info')
  })

  it('show("msg", "error") sets type to error', () => {
    toast.show('msg', 'error')
    expect(toast.toasts.value[0].type).toBe('error')
  })

  it('success creates type success', () => {
    toast.success('y')
    expect(toast.toasts.value[0].type).toBe('success')
  })

  it('error creates type error', () => {
    toast.error('e')
    expect(toast.toasts.value[0].type).toBe('error')
  })

  it('warn creates type warn', () => {
    toast.warn('w')
    expect(toast.toasts.value[0].type).toBe('warn')
  })

  it('info creates type info', () => {
    toast.info('i')
    expect(toast.toasts.value[0].type).toBe('info')
  })

  it('dedup: show("dup") twice quickly results in only 1 toast', () => {
    toast.show('dup')
    toast.show('dup')
    expect(toast.toasts.value).toHaveLength(1)
  })

  it('remove by id clears the toast', () => {
    toast.show('bye')
    const id = toast.toasts.value[0].id
    toast.remove(id)
    expect(toast.toasts.value).toHaveLength(0)
  })
})
