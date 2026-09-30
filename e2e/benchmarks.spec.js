// benchmarks.spec.js — page /benchmarks : l'index unifie s'affiche avec son compte.
const fs = require('fs');
const path = require('path');
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

// L'index servi en e2e est la fixture e2e/fixtures/bench_index.json (via BENCH_INDEX) :
// on lit le meme fichier pour connaitre le compte attendu sans le coder en dur.
const INDEX = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'fixtures', 'bench_index.json'), 'utf8'),
);
const N_ENTRIES = (INDEX.entries || []).length;

test('page /benchmarks : les tableaux de l\u2019index s\u2019affichent avec le compte d\u2019entrees', async ({
  page,
}) => {
  const guard = attachConsoleGuard(page);

  await page.goto('/benchmarks');

  await expect(page.getByRole('heading', { level: 1 })).toContainText('Benchmarks LLM');
  // « N entrees » est ecrit tel quel par benchmarks.py
  await expect(page.getByText(`${N_ENTRIES} entrees`)).toBeVisible();

  // les familles du registre ont chacune leur carte + tableau
  await expect(
    page.getByRole('heading', { name: /Index unifie des tests LLM/i }),
  ).toBeVisible();
  const tables = page.locator('main table');
  await expect.poll(() => tables.count(), { timeout: 10000 }).toBeGreaterThan(1);
  await expect(tables.first()).toBeVisible();

  // le compte annonce doit correspondre au nombre de tableaux/lignes non vides
  expect(N_ENTRIES).toBeGreaterThan(0);

  guard.assertClean('/benchmarks');
});
