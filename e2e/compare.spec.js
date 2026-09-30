// compare.spec.js — page /compare accessible, sur deux runs reels de la fixture.
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

test('page /compare : accessible et rend deux runs cote a cote', async ({ page, request }) => {
  const guard = attachConsoleGuard(page);

  // deux runs existent (fixtures copiees dans RUNS_DIR par run-tests.sh)
  const resp = await request.get('/api/runs?limit=500');
  expect(resp.ok()).toBeTruthy();
  const data = await resp.json();
  expect(Array.isArray(data.runs)).toBeTruthy();
  expect(data.runs.length).toBeGreaterThanOrEqual(2);

  const a = data.runs[1].run_id;
  const b = data.runs[0].run_id;

  await page.goto(`/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

  await expect(page.getByRole('heading', { level: 1 })).toContainText('Comparaison de deux runs');
  // au moins un prompt commun : les deux identites de run sont affichees
  await expect(page.locator('.id').first()).toBeVisible();
  await expect(page.getByText(a, { exact: false }).first()).toBeVisible();

  guard.assertClean('/compare');
});
