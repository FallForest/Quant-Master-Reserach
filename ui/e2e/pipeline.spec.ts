import { test, expect } from '@playwright/test'

test.describe('Removed routes', () => {
  test('redirects pipeline route to overview', async ({ page }) => {
    await page.goto('/pipeline')
    await page.waitForLoadState('domcontentloaded')
    await expect(page).toHaveURL(/\/$/)
    await expect(page.locator('body')).toContainText('总览')
  })

  test('redirects factor route to overview', async ({ page }) => {
    await page.goto('/factor')
    await page.waitForLoadState('domcontentloaded')
    await expect(page).toHaveURL(/\/$/)
    await expect(page.locator('body')).toContainText('总览')
  })
})
