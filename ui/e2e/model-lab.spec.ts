import { test, expect } from '@playwright/test'

test.describe('Model Lab Page', () => {
  test('loads model catalog', async ({ page }) => {
    await page.goto('/model-lab')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('模型工坊')
  })

  test('has category filter tabs', async ({ page }) => {
    await page.goto('/model-lab')
    await page.waitForLoadState('domcontentloaded')
    // Should have category filter buttons
    await expect(page.locator('body')).toContainText('全部')
  })
})
