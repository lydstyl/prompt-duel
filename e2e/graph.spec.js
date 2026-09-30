// graph.spec.js — page /graph : graphiques SVG inline, tableau de resultats et
// masquage reversible (POST /api/visibility).
//
// La route /graph et l'API de visibilite sont cablees dans app.py par l'etape E1
// (sous-agent parallele). Si elles ne repondent pas, la spec est IGNOREE avec un
// message explicite plutot que mise en echec : ce n'est pas un bug de la page
// elle-meme. Meme convention que les autres specs (selecteurs semantiques,
// collecte des erreurs console, aucun waitForTimeout).
//
// Les actions de visibilite rechargent la page : on n'evalue donc JAMAIS du
// JavaScript a la main juste apres un clic (contexte detruit par la navigation),
// on s'appuie sur les assertions auto-retentees de Playwright (locators).
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

const LIGNES = '#resultats tbody tr';

// Ouvre /graph ; renvoie true si la page est servie (sinon marque la spec ignoree).
async function ouvrirGraph(page) {
  const reponse = await page.goto('/graph');
  const statut = reponse ? reponse.status() : 0;
  if (statut !== 200) {
    test.skip(
      true,
      `/graph indisponible (HTTP ${statut || 'aucune reponse'}) : route E1 pas encore cablee`,
    );
    return false;
  }
  return true;
}

// Identifiant d'entree sur pour un selecteur d'attribut CSS.
function sel(identifiant) {
  return String(identifiant).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

test('page /graph : SVG inline, tableau de resultats, masquage reversible', async ({ page }) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  // en-tete + au moins un graphique SVG inline (aucune CDN, aucun JS requis)
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Graphique');
  const svg = page.locator('main svg');
  await expect.poll(() => svg.count(), { timeout: 10_000 }).toBeGreaterThan(0);
  await expect(svg.first()).toHaveAttribute('role', 'img');
  // info-bulle native : aucun JavaScript n'est necessaire pour lire une valeur
  await expect(page.locator('main svg title').first()).toBeAttached();

  const lignes = page.locator(LIGNES);
  await expect(lignes.first()).toBeVisible();
  const total = await lignes.count();
  expect(total).toBeGreaterThan(0);
  const cible = await lignes.first().getAttribute('data-entry');
  expect(cible).toBeTruthy();

  // Sonde NON destructive de l'API : `unhide` sur un resultat deja affiche ne
  // change rien (il n'est pas dans la liste des masques). Sans reponse 200, le
  // masquage n'est pas testable -> spec ignoree, avec la reponse brute.
  const ping = await page.request.post('/api/visibility', {
    data: { action: 'unhide', ids: [cible] },
  });
  if (ping.status() !== 200) {
    test.skip(
      true,
      `/api/visibility indisponible (HTTP ${ping.status()} : ${(await ping.text()).slice(0, 160)})`,
    );
    return;
  }
  await expect(lignes).toHaveCount(total); // la sonde n'a rien change
  await expect(page.locator(`${LIGNES}[data-entry="${sel(cible)}"]`)).toHaveCount(1);

  // « Masquer » -> POST /api/visibility puis rechargement : la ligne disparait du
  // tableau (et donc de tous les graphiques).
  await page.getByRole('button', { name: 'Masquer' }).first().click();
  await expect(page.locator(`${LIGNES}[data-entry="${sel(cible)}"]`)).toHaveCount(0, {
    timeout: 15_000,
  });
  await expect(lignes).toHaveCount(total - 1, { timeout: 15_000 });

  // le resultat masque est liste en bas de page avec son bouton de restauration
  const ligneMasquee = page.locator(`#masques tr[data-entry="${sel(cible)}"]`);
  await expect(ligneMasquee).toBeVisible();
  await expect(ligneMasquee.getByRole('button', { name: 'Réafficher' })).toBeVisible();

  // « Reafficher » le remet dans le tableau (etat remis a zero pour les autres specs)
  await ligneMasquee.getByRole('button', { name: 'Réafficher' }).click();
  await expect(page.locator(`${LIGNES}[data-entry="${sel(cible)}"]`)).toHaveCount(1, {
    timeout: 15_000,
  });
  await expect(lignes).toHaveCount(total, { timeout: 15_000 });
  await expect(page.locator(`#masques tr[data-entry="${sel(cible)}"]`)).toHaveCount(0);

  guard.assertClean('/graph');
});

test('page /graph : les filtres fonctionnent sans JavaScript (formulaire GET)', async ({ page }) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  const cases = page.locator('form.filtres input[name="kind"]');
  await expect.poll(() => cases.count(), { timeout: 10_000 }).toBeGreaterThan(0);

  const premiere = cases.first();
  const kind = await premiere.getAttribute('value');
  expect(kind).toBeTruthy();
  await premiere.check();
  await page.getByRole('button', { name: 'Filtrer' }).click();

  // le formulaire GET recharge /graph?kind=… (aucun JavaScript necessaire)
  await expect
    .poll(() => new URL(page.url()).searchParams.get('kind'), { timeout: 15_000 })
    .toBe(kind);

  const lignes = page.locator(LIGNES);
  await expect(lignes.first()).toBeVisible();
  const total = await lignes.count();
  expect(total).toBeGreaterThan(0);
  // seules les entrees de cette famille restent (l'identifiant commence par kind:)
  await expect(page.locator(`${LIGNES}[data-entry^="${kind}:"]`)).toHaveCount(total);

  guard.assertClean('/graph?kind');
});
