import { test, expect } from '@playwright/test'

test.describe('Pipeline Page', () => {
  test('loads pipeline page', async ({ page }) => {
    await page.goto('/pipeline')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('数据管道')
  })

  test('has run button', async ({ page }) => {
    await page.goto('/pipeline')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('一键更新')
  })
})
