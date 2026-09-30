// playwright.config.js — configuration de la suite e2e de prompt-duel.
//
// L'application n'est PAS demarree par Playwright : scripts/run-tests.sh lance
// l'app en LLM_DRY_RUN=1 sur un port de test, attend qu'elle reponde, puis
// invoque ce runner. On se contente donc de pointer baseURL sur l'app deja en
// cours (E2E_BASE_URL, sinon http://127.0.0.1:$E2E_PORT).
const { defineConfig } = require('playwright/test');

const PORT = process.env.E2E_PORT || '8799';
const baseURL = process.env.E2E_BASE_URL || `http://127.0.0.1:${PORT}`;

module.exports = defineConfig({
  // les specs vivent a cote de ce fichier (e2e/*.spec.js)
  testDir: __dirname,
  testMatch: '**/*.spec.js',
  // un seul worker : les specs partagent la meme instance d'app et le meme etat
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  use: {
    baseURL,
    headless: true,
    actionTimeout: 15_000,
    navigationTimeout: 20_000,
    trace: 'off',
    video: 'off',
    screenshot: 'off',
  },
});
