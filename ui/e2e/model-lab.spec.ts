import { test, expect } from '@playwright/test'

test.describe('Model Lab Page', () => {
  test('loads decision center shell', async ({ page }) => {
    await page.goto('/model-lab')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('实盘决策中心')
  })

  test('shows deployment decision content', async ({ page }) => {
    await page.goto('/model-lab')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.locator('body')).toContainText('Deployment Principle')
    await expect(page.locator('body')).toContainText('最优候选与部署顺序')
  })
})
