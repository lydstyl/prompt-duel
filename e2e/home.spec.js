// home.spec.js — page d'accueil : formulaire + catalogue de prompts.
const fs = require('fs');
const path = require('path');
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

// Le catalogue est celui de l'app (prompts.json a la racine du depot) : on lit
// le fichier reel pour ne dependre d'aucune donnee codee en dur.
const CATALOG = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', 'prompts.json'), 'utf8'),
);
const N_PROMPTS = CATALOG.prompts.length;

test("page d'accueil : le formulaire se charge et le catalogue est liste", async ({ page }) => {
  const guard = attachConsoleGuard(page);

  await page.goto('/');

  // titre + formulaire principal
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Prompt Duel');
  await expect(page.locator('#launch')).toBeVisible();
  await expect(page.locator('#stop')).toBeVisible();
  await expect(page.locator('#settings')).toBeVisible();
  await expect(page.locator('#set-temp')).toBeVisible();

  // le badge LLM doit basculer en mode simule (dry run) une fois /api/state recu
  await expect(page.locator('#llm')).toContainText('DRY RUN', { timeout: 15000 });

  // catalogue de prompts : toutes les cases sont rendues, groupees par <details>
  await expect(page.locator('#prompts input[type=checkbox]')).toHaveCount(N_PROMPTS);
  await expect(page.locator('#prompts details.grp').first()).toBeVisible();
  await expect(page.locator('#count')).toContainText(`/ ${N_PROMPTS} sélectionné(s)`);

  // section « tests » rendue elle aussi
  await expect(page.locator('#tests details.grp').first()).toBeVisible();

  guard.assertClean("/");
});
