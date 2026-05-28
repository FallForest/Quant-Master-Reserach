import { test, expect } from '@playwright/test'

test.describe('Browser Page', () => {
  test('loads stock list', async ({ page }) => {
    await page.goto('/browser')
    await page.waitForLoadState('domcontentloaded')
    // Wait for page content to render
    await expect(page.locator('body')).toContainText('数据浏览', { timeout: 10000 })
  })

  test('has search input', async ({ page }) => {
    await page.goto('/browser')
    await page.waitForLoadState('domcontentloaded')
    const search = page.locator('input[type="text"]')
    await expect(search).toBeVisible({ timeout: 10000 })
  })
})
