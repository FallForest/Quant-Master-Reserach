import { test, expect } from '@playwright/test'

const routes = [
  { path: '/', title: '总览' },
  { path: '/browser', title: '数据浏览' },
  { path: '/model-lab', title: '模型工坊' },
  { path: '/model-performance', title: '模型绩效' },
  { path: '/stock-select', title: '模型选股' },
  { path: '/experiments', title: '实验管理' },
  { path: '/strategy-lab', title: '策略工坊' },
  { path: '/backtest', title: '策略回测' },
  { path: '/portfolio', title: '持仓分析' },
  { path: '/optimizer', title: '组合优化' },
  { path: '/attribution', title: '收益归因' },
]

test.describe('Navigation', () => {
  for (const route of routes) {
    test(`can navigate to ${route.path}`, async ({ page }) => {
      await page.goto(route.path)
      await page.waitForLoadState('domcontentloaded')
      // Page should not show error
      await expect(page.locator('body')).not.toContainText('404', { timeout: 10000 })
      await expect(page.locator('body')).not.toContainText('error', { timeout: 10000 })
    })
  }

  test('sidebar shows all navigation items', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    // Check that sidebar contains navigation group titles
    await expect(page.locator('aside')).toContainText('概览')
    await expect(page.locator('aside')).toContainText('数据')
    await expect(page.locator('aside')).toContainText('模型')
    await expect(page.locator('aside')).toContainText('策略')
    await expect(page.locator('aside')).toContainText('分析')
  })
})
