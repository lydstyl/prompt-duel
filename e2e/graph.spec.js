// graph.spec.js — page /graph parcourue en DEUX ETAPES (v1.6.3).
//
// Etape 1 (`/graph`) : le catalogue des reglages, une case a cocher par variante
//   (modele + contexte/MTP/thinking…), la regle « coche de 2 a 6 » ecrite en clair,
//   un bouton « Comparer » — et AUCUN graphique.
// Etape 2 (`/graph?sel=<cle>&sel=<cle>`) : uniquement les graphiques des reglages
//   coches, un lien « modifier la selection » pour revenir au choix, puis le
//   panneau « affiner » et le tableau des resultats (masquage reversible).
//
// Tout se fait en GET, sans JavaScript applicatif : les specs pilotent donc de vrais
// formulaires. Si /graph ne repond pas, la spec est IGNOREE avec un message explicite
// plutot que mise en echec. Meme convention que les autres specs (selecteurs
// semantiques, collecte des erreurs console, aucun waitForTimeout).
//
// Les actions de visibilite rechargent la page : on n'evalue donc JAMAIS du
// JavaScript a la main juste apres un clic (contexte detruit par la navigation),
// on s'appuie sur les assertions auto-retentees de Playwright (locators).
const { test, expect } = require('playwright/test');
const { attachConsoleGuard } = require('./helpers');

const LIGNES = '#resultats tbody tr';
const CASES_SEL = '#selection input[name="sel"]';

// Identifiant d'entree sur pour un selecteur d'attribut CSS.
function sel(identifiant) {
  return String(identifiant).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

// Ouvre /graph ; renvoie true si la page est servie (sinon marque la spec ignoree).
async function ouvrirGraph(page, url = '/graph') {
  const reponse = await page.goto(url);
  const statut = reponse ? reponse.status() : 0;
  if (statut !== 200) {
    test.skip(
      true,
      `/graph indisponible (HTTP ${statut || 'aucune reponse'}) : route pas encore cablee`,
    );
    return false;
  }
  return true;
}

// Coche les `n` premiers reglages de l'etape 1, puis valide « Comparer ».
// Renvoie les cles cochees (lues dans le formulaire : aucune donnee codee en dur).
async function comparer(page, n = 2) {
  await expect(page.locator('#selection')).toBeVisible();
  const cases = page.locator(CASES_SEL);
  const total = await cases.count();
  expect(total).toBeGreaterThanOrEqual(n);
  const cles = [];
  for (let i = 0; i < n; i += 1) {
    cles.push(await cases.nth(i).getAttribute('value'));
    await cases.nth(i).check();
  }
  await page.getByRole('button', { name: 'Comparer' }).click();
  await expect(page.locator('main svg').first()).toBeVisible({ timeout: 15_000 });
  return cles;
}

test('etape 1 : choisir de 2 a 6 reglages, sans aucun graphique', async ({ page }) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  // en-tete + carte de selection : la regle 2-6 est ecrite en clair
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Graphique');
  const selection = page.locator('#selection');
  await expect(selection).toBeVisible();
  await expect(selection.getByRole('heading')).toContainText('Comparer des réglages');
  await expect(selection).toContainText('2 au minimum');
  await expect(selection).toContainText('6 au maximum');
  await expect(selection).toContainText('puis « Comparer »');

  // le catalogue : une case par reglage, avec points et date du dernier resultat
  const cases = page.locator(CASES_SEL);
  await expect.poll(() => cases.count(), { timeout: 10_000 }).toBeGreaterThan(1);
  await expect(cases.first()).toHaveAttribute('value', /.+/);
  const premiere = page.locator('#selection .selection-list label').first();
  await expect(premiere).toContainText('point');
  await expect(premiere).toContainText('dernier le');

  // AUCUN graphique a l'etape 1 (c'est le principe de la vue par defaut)
  await expect(page.locator('main svg')).toHaveCount(0);
  // ... mais le reste de la page reste accessible, replie (details natif, sans JS)
  await expect(page.locator('details#tous-resultats')).toBeAttached();

  guard.assertClean('/graph (etape 1)');
});

test('etape 2 : seuls les reglages coches sont traces, masquage reversible', async ({ page }) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  const cles = await comparer(page, 2);

  // l'adresse de l'etape 2 porte les deux cles cochees (GET, partageable)
  const params = new URL(page.url()).searchParams;
  expect(params.getAll('sel')).toEqual(cles);

  // les graphiques sont la : SVG inline (role=img) + info-bulle native
  const svg = page.locator('main svg');
  await expect.poll(() => svg.count(), { timeout: 10_000 }).toBeGreaterThan(0);
  await expect(svg.first()).toHaveAttribute('role', 'img');
  await expect(page.locator('main svg title').first()).toBeAttached();

  // la selection est rappelee, et seules ces series apparaissent dans les legendes
  await expect(page.locator('#selection')).toContainText('Comparaison de 2 réglages');
  const seriesTracees = await page
    .locator('.chart-legend [data-serie]')
    .evaluateAll((noeuds) => Array.from(new Set(noeuds.map((n) => n.getAttribute('data-serie')))));
  expect(seriesTracees.length).toBeGreaterThan(0);
  for (const cle of seriesTracees) {
    expect(cles).toContain(cle);
  }

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
  // tableau (et donc de tous les graphiques), sans perdre la comparaison en cours.
  await page.getByRole('button', { name: 'Masquer' }).first().click();
  await expect(page.locator(`${LIGNES}[data-entry="${sel(cible)}"]`)).toHaveCount(0, {
    timeout: 15_000,
  });
  await expect(lignes).toHaveCount(total - 1, { timeout: 15_000 });
  expect(new URL(page.url()).searchParams.getAll('sel')).toEqual(cles);

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

  // « modifier la selection » ramene a l'etape 1, cases conservees
  await page.getByRole('link', { name: /modifier la sélection/ }).click();
  await expect(page.locator('#selection')).toContainText('Comparer des réglages', {
    timeout: 15_000,
  });
  await expect(page.locator('main svg')).toHaveCount(0);
  await expect(page.locator('#selection input[name="sel"]:checked')).toHaveCount(2);

  guard.assertClean('/graph (etape 2)');
});

test('bornes : 1 reglage renvoie au choix, plus de 6 tronque, cle inconnue signalee', async ({
  page,
}) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  const cles = await page
    .locator(CASES_SEL)
    .evaluateAll((noeuds) => noeuds.map((n) => n.getAttribute('value')));
  expect(cles.length).toBeGreaterThanOrEqual(7);

  // 1 reglage -> retour a l'etape 1, message clair, aucun graphique
  await ouvrirGraph(page, `/graph?sel=${encodeURIComponent(cles[0])}`);
  await expect(page.locator('#selection')).toContainText('au moins 2');
  await expect(page.locator('main svg')).toHaveCount(0);

  // 7 reglages -> les 6 premiers sont gardes, le 7e est nomme dans un bandeau
  const sept = `/graph?${cles.slice(0, 7).map((c) => `sel=${encodeURIComponent(c)}`).join('&')}`;
  await ouvrirGraph(page, sept);
  await expect(page.locator('.banner')).toContainText('6 réglages au maximum');
  await expect(page.locator('.banner')).toContainText(cles[6]);
  const tracees = await page
    .locator('.chart-legend [data-serie]')
    .evaluateAll((noeuds) => Array.from(new Set(noeuds.map((n) => n.getAttribute('data-serie')))));
  expect(tracees.length).toBeLessThanOrEqual(6);

  // cle inconnue -> signalee, jamais appliquee (2 vraies cles + 1 inconnue)
  const inconnue = 'reglage-qui-nexiste-pas';
  const melange = `/graph?sel=${encodeURIComponent(cles[0])}&sel=${encodeURIComponent(cles[1])}`
    + `&sel=${inconnue}`;
  await ouvrirGraph(page, melange);
  await expect(page.locator('.banner')).toContainText('inconnu');
  await expect(page.locator('.banner')).toContainText(inconnue);
  await expect(page.locator('main svg').first()).toBeVisible();

  // ... et si TOUTES les cles sont inconnues : etape 1 + message, jamais de 500
  await ouvrirGraph(page, '/graph?sel=inconnu-a&sel=inconnu-b');
  await expect(page.locator('main svg')).toHaveCount(0);
  await expect(page.locator('.banner.err')).toBeVisible();

  guard.assertClean('/graph (bornes)');
});

test('etape 2 : les filtres marchent sans JavaScript et gardent la selection', async ({ page }) => {
  const guard = attachConsoleGuard(page);
  if (!(await ouvrirGraph(page))) return;

  const cles = await comparer(page, 2);

  // panneau « affiner » : les familles presentes dans l'index sont listees
  const affiner = page.locator('#affiner');
  const cases = affiner.locator('input[name="kind"]');
  await expect.poll(() => cases.count(), { timeout: 10_000 }).toBeGreaterThan(0);

  // on filtre sur une famille REELLEMENT presente dans la comparaison en cours
  // (l'identifiant d'entree commence par « <famille>: ») : le panneau liste toutes
  // les familles de l'index, pas seulement celles des reglages coches.
  const lignes = page.locator(LIGNES);
  await expect(lignes.first()).toBeVisible();
  const totalAvant = await lignes.count();
  const kind = (await lignes.first().getAttribute('data-entry')).split(':')[0];
  expect(kind).toBeTruthy();
  const premiere = affiner.locator(`input[name="kind"][value="${sel(kind)}"]`);
  await expect(premiere).toHaveCount(1);
  await premiere.check();
  await affiner.getByRole('button', { name: 'Filtrer' }).click();

  // le formulaire GET recharge /graph?sel=…&kind=… : la selection est conservee
  await expect
    .poll(() => new URL(page.url()).searchParams.get('kind'), { timeout: 15_000 })
    .toBe(kind);
  await expect
    .poll(() => new URL(page.url()).searchParams.getAll('sel'), { timeout: 15_000 })
    .toEqual(cles);

  // seules les entrees de cette famille restent, et les graphiques sont toujours la
  await expect(lignes.first()).toBeVisible();
  const total = await lignes.count();
  expect(total).toBeGreaterThan(0);
  expect(total).toBeLessThanOrEqual(totalAvant);
  await expect(page.locator(`${LIGNES}[data-entry^="${kind}:"]`)).toHaveCount(total);
  await expect(page.locator('main svg').first()).toBeVisible();

  guard.assertClean('/graph?sel&kind');
});
