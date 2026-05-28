import { test, expect } from '@playwright/test'

test.describe('Overview Page', () => {
  test('loads and displays content', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    // Should have the page title
    await expect(page.locator('body')).toContainText('总览')
    // Should have stat cards area
    await expect(page.locator('body')).toContainText('股票总数')
  })

  test('has quick action links', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('浏览数据')
    await expect(page.locator('body')).toContainText('模型工坊')
  })
})
