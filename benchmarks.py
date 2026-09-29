#!/usr/bin/env python3
"""benchmarks.py — page /benchmarks de Prompt Duel : rend l'index unifie des tests.

Lit l'index JSON produit par bench_index.py (vault en priorite, copie locale en secours)
et affiche une page HTML autonome : duels HTML, vitesse/VRAM, batteries de taches,
HumanEval, remplissages de contexte.

Aucune dependance : stdlib uniquement, comme le reste de l'app.
"""
import html
import json
import os
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordre de priorite des sources d'index
INDEX_CANDIDATES = [
    os.environ.get("BENCH_INDEX") or "",
    "/home/gab/NAS/AgentsMirror/vaults/personnel/documents/llm-benchmarks/index.json",
    os.path.join(APP_DIR, "bench_index.json"),
]
EMPTY_HINT = ("Aucun index trouve. Genere-le depuis ~/apps/prompt-duel :<br>"
              "<code>python3 bench_index.py</code>")

CSS = """
:root{--bg:#0d0f14;--bg2:#141821;--fg:#e6e9ef;--dim:#8b93a5;--line:#242a36;--acc:#4da3ff;--ok:#4ddb8b;--err:#ff6b6b;--warn:#ffc85c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--bg2);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:14px;margin:0}
.dim{color:var(--dim);font-size:12.5px}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
.navlink{color:var(--acc);text-decoration:none;font-size:12.5px;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
main{padding:18px;max-width:1500px;margin:0 auto}
section.card{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.card>h2{margin-bottom:10px;display:flex;gap:10px;align-items:baseline}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#171c26}
.tag{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px;color:var(--dim)}
.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}
.scroll{overflow-x:auto}
code{background:#0a0c11;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
footer{padding:14px 18px;color:var(--dim);font-size:12px;border-top:1px solid var(--line)}
"""


def _load_index():
    for path in INDEX_CANDIDATES:
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                data["_source"] = path
                return data
            except Exception as exc:
                return {"error": f"{path} illisible : {exc}"}
    return {"error": None}


def _esc(v):
    return html.escape("" if v is None else str(v))


def _num(v, dec=1, suffix=""):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    return f"{f:,.{dec}f}".replace(",", " ") + suffix


def _date(v):
    if not v:
        return "-"
    s = str(v)
    try:
        return datetime.fromisoformat(s).strftime("%d/%m %H:%M")
    except ValueError:
        return s[:16]


def _rows(entries, kind, model_keys, columns, row_fn):
    return [e for e in entries if e.get("kind") == kind]


def _table(headers, rows, cls=""):
    if not rows:
        return '<p class="dim">aucun resultat</p>'
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(c for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _kind_of_score(score, total):
    try:
        if float(score) >= float(total):
            return '<span class="ok">complet</span>'
    except (TypeError, ValueError):
        pass
    return '<span class="warn">partiel</span>'


def render_body(data):
    if data.get("error"):
        return f'<section class="card"><h2>Index</h2><p class="err">{_esc(data["error"])}</p></section>'
    entries = data.get("entries") or []
    if not entries:
        return f'<section class="card"><h2>Index</h2><p class="dim">{EMPTY_HINT}</p></section>'

    out = []
    gen = _date(data.get("generated_at"))
    out.append(f'<section class="card"><h2>Index unifie des tests LLM</h2>'
               f'<p class="dim">genere le {gen} par <code>bench_index.py</code> &middot; '
               f'{len(entries)} entrees &middot; source : <code>{_esc(data.get("_source"))}</code></p>'
               f'<p class="dim">Le vault est la source de verite (notes + <code>documents/llm-benchmarks/</code>) ; '
               f'cette page en est la vue visuelle. Un « run » de duel = un modele, un contexte, un jeu de reglages.</p>'
               f'</section>')

    # 1) Duels HTML (prompt-duel)
    duels = sorted([e for e in entries if e.get("kind") == "duel"],
                   key=lambda e: e.get("date") or "", reverse=True)
    if duels:
        rows = []
        for e in duels:
            rows.append([
                f'<td>{_date(e.get("date"))}</td>',
                f'<td>{_esc(e.get("model"))}</td>',
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("temperature"),1)}</td>',
                f'<td>{"oui" if e.get("thinking") else "non"}</td>',
                f'<td class="n">{_esc(e.get("prompts_ok"))}/{_esc(e.get("prompts_total"))}</td>',
                f'<td class="n">{_num(e.get("duration_s"),0," s")}</td>',
                f'<td class="n">{_num(e.get("tokens_out"),0)}</td>',
                f'<td class="n">{_num(e.get("tok_s"))}</td>',
                f'<td><a href="/compare">comparer</a> &middot; <a href="{_esc(e.get("run_url"))}">fichiers</a></td>',
            ])
        out.append('<section class="card"><h2>Duels HTML <span class="dim">'
                   '32+ prompts canvas/3D/jeu &rarr; une page HTML par prompt</span></h2>'
                   + _table(["date", "modele", "ctx", "temp", "reasoning", "prompts", "duree",
                             "tokens sortie", "tok/s", ""], rows) + '</section>')

    # 2) Vitesse & VRAM
    speeds = sorted([e for e in entries if e.get("kind") == "speed"],
                    key=lambda e: (e.get("model") or "", e.get("ctx") or 0))
    if speeds:
        rows = []
        for e in speeds:
            rows.append([
                f'<td>{_esc(e.get("model"))}</td>',
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("tg"))}</td>',
                f'<td class="n">{_num(e.get("pp"),0)}</td>',
                f'<td class="n">{_esc(e.get("vram"))}</td>',
                f'<td class="dim">{_esc(e.get("note") or "")}</td>',
            ])
        out.append('<section class="card"><h2>Vitesse &amp; VRAM <span class="dim">'
                   'llama-server .224 &middot; PP = prefill, TG = generation (KV q8_0)</span></h2>'
                   + _table(["modele", "ctx", "TG t/s", "PP t/s", "VRAM 5060Ti/5080", "note"], rows) + '</section>')

    # 3) Batteries de petites taches
    bat = sorted([e for e in entries if e.get("kind") == "battery"],
                 key=lambda e: (e.get("model") or "", e.get("mode") or ""))
    if bat:
        rows = []
        for e in bat:
            fails = e.get("fails") or []
            rows.append([
                f'<td>{_esc(e.get("model"))}</td>',
                f'<td class="tag">{_esc(e.get("mode"))}</td>',
                f'<td class="n">{_esc(e.get("score"))}/{_esc(e.get("tasks"))}</td>',
                f'<td>{_kind_of_score(e.get("score"), e.get("tasks"))}</td>',
                f'<td class="n">{_num(e.get("lat_median"),2," s")}</td>',
                f'<td class="n">{_num(e.get("lat_max"),1," s")}</td>',
                f'<td class="dim">{_esc(", ".join(fails)) if fails else "aucun"}</td>',
            ])
        out.append('<section class="card"><h2>Batteries de petites taches <span class="dim">'
                   '13 taches FR reelles, scoring verifie par code</span></h2>'
                   + _table(["modele", "mode", "score", "", "latence med", "latence max", "echecs"], rows)
                   + '<p class="dim">« mode » = reasoning coupe (rapide) ou reasoning medium. '
                     'Les 2 echecs communs en mode rapide (prorata, jour de la semaine) sont une limite du '
                     'mode sans reflexion, pas d\'un modele.</p></section>')

    # 4) HumanEval
    he = sorted([e for e in entries if e.get("kind") == "humaneval"],
                key=lambda e: (-(e.get("total") or 0), -(e.get("score_pct") or 0)))
    if he:
        rows = []
        for e in he:
            rows.append([
                f'<td>{_esc(e.get("model"))}</td>',
                f'<td class="dim">{_esc(e.get("config") or "")}</td>',
                f'<td class="n">{_esc(e.get("passed"))}/{_esc(e.get("total"))}</td>',
                f'<td class="n">{_num(e.get("score_pct"))} %</td>',
                f'<td class="n">{_num(e.get("avg_latency_s"),2," s")}</td>',
                f'<td class="n">{_num(e.get("wall_s"),0," s")}</td>',
                f'<td class="dim">{_date(e.get("date"))}</td>',
            ])
        out.append('<section class="card"><h2>HumanEval <span class="dim">'
                   'sous-ensemble 0-49, extraction validee 5/5 avant le run, reasoning coupe</span></h2>'
                   + _table(["modele", "config", "passes", "score", "latence/probleme", "temps mural", "date"], rows)
                   + '<p class="dim">Tri : jeux complets (164 problemes) d\'abord, puis score. '
                     'Les sous-ensembles courts (5, 23 ou 38 problemes) ne se comparent pas aux runs complets.</p>'
                   + '</section>')

    # 5) Contexte (remplissages reels)
    ctxs = sorted([e for e in entries if e.get("kind") == "context"],
                  key=lambda e: -(e.get("ctx") or 0))
    if ctxs:
        rows = []
        for e in ctxs:
            ok = e.get("needle")
            verdict = ('<span class="ok">valide</span>' if ok else
                       '<span class="err">echec</span>' if e.get("status") == "error" else
                       '<span class="warn">-</span>')
            rows.append([
                f'<td>{_esc(e.get("model"))}</td>',
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("tokens"),0)}</td>',
                f'<td>{verdict}</td>',
                f'<td class="n">{_num(e.get("prefill_tps"),0)}</td>',
                f'<td class="n">{_esc(e.get("peak_vram"))}</td>',
                f'<td class="dim">{_esc(e.get("note") or "")}</td>',
            ])
        out.append('<section class="card"><h2>Contexte long <span class="dim">'
                   'remplissage reel + needle a 85 % de profondeur (jamais un simple /health)</span></h2>'
                   + _table(["modele", "ctx", "tokens ingeres", "needle", "prefill t/s", "pic VRAM", "note"], rows)
                   + '</section>')

    # 6) Notes du vault (liens sortants) et artefacts
    notes = data.get("vault_notes") or []
    if notes:
        items = "".join(f'<li><code>{_esc(n.get("path"))}</code> — {_esc(n.get("title"))}</li>' for n in notes)
        out.append(f'<section class="card"><h2>Notes du vault</h2><ul class="dim" style="margin:6px 0 0 0">{items}</ul>'
                   f'<p class="dim">Chemin du vault sur le .224 : '
                   f'<code>/home/gab/NAS/AgentsMirror/vaults/personnel/wiki/casquettes/developpeur/llm-benchmarks/</code></p>'
                   f'</section>')
    return "".join(out)


def render_benchmarks_page():
    data = _load_index()
    if not data.get("error") and not data.get("entries"):
        data = dict(data)
    body = render_body(data)
    gen = _date(data.get("generated_at"))
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks LLM — Prompt Duel</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Benchmarks LLM <span class="dim">&middot; index unifie des tests locaux</span></h1>
  <div class="spacer" style="flex:1"></div>
  <a class="navlink" href="/">&#8592; Prompt Duel</a>
  <a class="navlink" href="/compare">Comparaison de runs</a>
  <span class="dim">index {gen}</span>
</header>
<main>{body}</main>
<footer>Prompt Duel 224 &middot; page <code>/benchmarks</code> &middot; index regenere par
<code>python3 bench_index.py</code> (vault = source de verite, copie locale en secours).</footer>
</body></html>""".encode("utf-8")


if __name__ == "__main__":
    import sys
    sys.stdout.write(render_benchmarks_page().decode("utf-8"))
