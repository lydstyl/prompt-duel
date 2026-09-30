// helpers.js — utilitaires partages par les specs e2e.
//
// Le seul point delicat : la page d'accueil declenche la requete Chrome pour
// /favicon.ico. Le serveur (stdlib) repond 404 sans favicon : Chromium journalise
// alors « Failed to load resource ... 404 ». Ce n'est pas une erreur applicative,
// on la filtre explicitement (le but est de detecter les VRAIES erreurs JS).
const { expect } = require('playwright/test');

function isFaviconNoise(msg) {
  const text = msg.text() || '';
  let url = '';
  try {
    const loc = typeof msg.location === 'function' ? msg.location() : null;
    url = (loc && loc.url) || '';
  } catch (e) {
    url = '';
  }
  const blob = `${text} ${url}`.toLowerCase();
  return blob.includes('favicon');
}

// Branche la collecte des erreurs console + exceptions de page sur `page`.
// Renvoie { errors, assertClean(label) }.
function attachConsoleGuard(page) {
  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    if (isFaviconNoise(msg)) return;
    errors.push(msg.text());
  });
  page.on('pageerror', (err) => {
    errors.push('pageerror: ' + (err && err.message ? err.message : String(err)));
  });
  return {
    errors,
    assertClean(label) {
      expect(
        errors,
        `erreurs console inattendues sur ${label} : ${errors.join(' | ')}`,
      ).toEqual([]);
    },
  };
}

module.exports = { attachConsoleGuard };
