// duel-run.spec.js — un vrai duel en LLM_DRY_RUN=1 : lancement, apparition du run,
// evolution du statut jusqu'a la fin. C'est le seul test qui ECRIT dans RUNS_DIR ;
// run-tests.sh pointe RUNS_DIR sur un dossier temporaire, jamais sur runs/ du depot.
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

async function runCount(request) {
  const resp = await request.get('/api/runs?limit=500');
  const data = await resp.json();
  return (data.runs || []).length;
}

test('duel dry-run : lancement, run liste et statut jusqu\u2019au bout', async ({ page, request }) => {
  const guard = attachConsoleGuard(page);

  await page.goto('/');
  // s'assurer que /api/state est passe une fois (badge dry) : les boutons se cablent alors
  await expect(page.locator('#llm')).toContainText('DRY RUN', { timeout: 15000 });

  const before = await runCount(request);

  // cocher le premier prompt puis lancer
  await page.locator('#prompts input[type=checkbox]').first().check();
  const launch = page.locator('#launch');
  await expect(launch).toBeEnabled();
  await launch.click();

  // l'UI montre le run courant
  await expect(page.locator('#runinfo')).toContainText('Dossier du run', { timeout: 20000 });

  // le run apparait dans la liste des runs servie par l'app
  await expect
    .poll(() => runCount(request), {
      timeout: 40000,
      message: 'le nouveau run doit apparaitre dans /api/runs',
    })
    .toBeGreaterThan(before);

  // le statut evolue et se termine (dry run : un seul prompt, ~3 s)
  await expect
    .poll(
      async () => {
        const resp = await request.get('/api/state');
        const state = await resp.json();
        return state.run ? state.run.status : 'none';
      },
      { timeout: 45000, message: 'le run doit atteindre un statut terminal' },
    )
    .toMatch(/finished|error|stopped/);

  const state = await (await request.get('/api/state')).json();
  expect(state.run.is_running).toBeFalsy();
  expect(state.run.dry_run).toBeTruthy();

  guard.assertClean('/ (duel)');
});
