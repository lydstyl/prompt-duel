#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charts.py — graphiques en SVG inline pour prompt-duel (v1.6).

Contraintes tenues ici :
- **stdlib pure** : aucune dépendance, aucun CDN, aucun JavaScript obligatoire ;
- SVG **autonome** (`viewBox`, `width="100%"`, `max-width`, police héritée) et
  **thème sombre** (couleurs claires, grille discrète) : l'app est en dark ;
- info-bulles via `<title>` natif (aucun JS) : série, groupe, valeur + unité,
  durée réelle lisible, date, statut ;
- fonctions **pures et déterministes** : aucune I/O, aucune horloge, aucun
  aléatoire — deux appels identiques donnent la même chaîne ;
- **jamais d'exception** et jamais de NaN/Infinity dans le SVG : les données
  douteuses (None, texte, valeur non finie, métrique absente) sont ignorées.

Entrée attendue (contrat §6, sortie de `metrics.series_for_chart`) ::

    {"metric": {"key", "label", "unit", "better"},
     "groups": [{"key", "label"}],
     "series": [{"key", "label", "hidden": bool,
                 "points": [{"group", "value", "duration_s",
                             "entry_id", "date", "status"}]}]}

Tout texte injecté (série, groupe, unité, statut) passe par `html.escape`.
"""

import html
import math
import re

__all__ = ["grouped_bars", "bars", "duration_chart", "legend", "PALETTE"]

# ---------------------------------------------------------------------------
# Constantes de rendu
# ---------------------------------------------------------------------------

# 10 teintes lisibles sur fond sombre (les 5 premières reprennent l'app).
PALETTE = (
    "#4da3ff",  # bleu (--acc)
    "#b58cff",  # violet
    "#2ecc71",  # vert (--ok)
    "#f39c12",  # orange (--warn)
    "#e74c3c",  # rouge (--err)
    "#1abc9c",  # turquoise
    "#e67e22",  # ambre
    "#9b59b6",  # magenta
    "#00bcd4",  # cyan
    "#f1c40f",  # jaune
)

C_FG = "#e6e8ee"     # texte principal (--fg)
C_DIM = "#98a0b3"    # texte secondaire (--dim)
C_GRID = "#2a2f3a"   # grille (--line)
C_ZERO = "#4a5266"   # axe du zéro

CHAR_W = 6.2         # largeur moyenne d'un caractère en font-size 11 (estimation)
CHAR_W_MAJ = 7.2     # majuscules et chiffres sont plus larges
FS_TITLE = 12        # titre d'axe
FS_LABEL = 11        # libellés d'échelle / groupes
FS_VALUE = 10        # valeur au-dessus des barres
DEFAULT_W = 980      # largeur de référence (le SVG reste fluide)
_H_MESSAGE = 120     # hauteur des SVG « aucune donnée »
_MARGIN_TOP = 30
_MARGIN_BOTTOM = 44
_MARGIN_RIGHT = 12
_MAX_BARS = 600      # garde-fou : nombre max de barres réellement dessinées
_MAX_SERIES = 60     # garde-fou : nombre max de séries alignées dans un groupe


# ---------------------------------------------------------------------------
# Petits utilitaires purs
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    """Échappe tout texte injecté dans le SVG (série, groupe, unité, statut…)."""
    return html.escape("" if value is None else str(value), quote=True)


def _num(raw):
    """Convertit en float fini, sinon None (jamais d'exception).

    Accepte les nombres et les chaînes du genre « 11,7 » ou « 11.7 / 11.9 Go »
    (on garde le premier nombre). Refuse bool, NaN et ±Infinity.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, str):
        txt = raw.replace("\u202f", "").replace("\xa0", " ").strip()
        m = re.match(r"^([+-]?\d+(?:[.,]\d+)?)", txt)
        if not m:
            return None
        try:
            val = float(m.group(1).replace(",", "."))
        except ValueError:
            return None
        return val if math.isfinite(val) else None
    return None


def _num_pur(raw):
    """Nombre fini « strict » : refuse « 12 min » (utile pour les durées)."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, str):
        txt = raw.strip().replace("\xa0", "").replace("\u202f", "").replace(",", ".")
        try:
            val = float(txt)
        except ValueError:
            return None
        return val if math.isfinite(val) else None
    return None


def _fmt_num(val) -> str:
    """Nombre en français : virgule décimale, espace comme séparateur de milliers."""
    if val is None or not math.isfinite(float(val)):
        return "—"
    val = float(val)
    a = abs(val)
    if a >= 1000:
        dec = 0 if val == int(val) else 1      # 1 420,5 t/s : on ne perd pas la décimale
    elif a >= 100:
        dec = 1
    elif a >= 10:
        dec = 2
    else:
        dec = 3
    txt = f"{val:,.{dec}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    txt = txt.replace(",", " ").replace(".", ",")
    return "0" if txt in ("-0", "") else txt


def _fmt_value(val, unit="") -> str:
    """Valeur lisible + unité (« 12,3 t/s ») ; unité absente acceptée."""
    txt = _fmt_num(val)
    unit = "" if unit is None else str(unit).strip()
    return f"{txt} {unit}" if unit else txt


def _fmt_duration(sec, short=False) -> str:
    """Durée en français lisible : « 12 min 35 s », « 1 h 02 min », « 45 s »."""
    val = _num_pur(sec)
    if val is None:
        return "—"
    sign = "-" if val < 0 else ""
    val = abs(val)
    if val < 60:
        if short or val == int(val) or val >= 10:
            return f"{sign}{int(round(val))} s"
        return f"{sign}{_fmt_num(round(val, 1))} s"
    total = int(round(val))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if short:
        if hours:
            return f"{sign}{hours} h" + (f" {minutes:02d}" if minutes else "")
        return f"{sign}{minutes} min"
    if hours:
        parts = [f"{hours} h", f"{minutes:02d} min"]
        if secs:
            parts.append(f"{secs:02d} s")
    else:
        parts = [f"{minutes} min"]
        if secs:
            parts.append(f"{secs} s")
    return sign + " ".join(parts)


def _fmt_date(raw) -> str:
    """Date lisible (30/09/2026 11:46) — pure, ne lit jamais l'horloge."""
    txt = "" if raw is None else str(raw).strip()
    if not txt:
        return "—"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?", txt)
    if not m:
        return txt
    out = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    if m.group(4):
        out += f" {m.group(4)}:{m.group(5)}"
    return out


_STATUS_FR = {
    "ok": "ok",
    "error": "échec",
    "erreur": "échec",
    "fail": "échec",
    "failed": "échec",
    "partial": "partiel",
    "partiel": "partiel",
    "skipped": "ignoré",
}


def _fmt_status(raw) -> str:
    txt = "" if raw is None else str(raw).strip()
    if not txt:
        return "—"
    return _STATUS_FR.get(txt.lower(), txt)


def _largeur(text) -> float:
    """Largeur estimée d'un texte en font-size 11, sans métrique de police."""
    return sum(CHAR_W_MAJ if (c.isupper() or c.isdigit()) else CHAR_W for c in str(text))


def _trunc_px(text, max_px, ell="…") -> str:
    """Tronque un libellé à la place réellement disponible.

    La coupe tombe toujours sur une frontière de caractère (jamais au milieu
    d'un glyphe) : on retire des caractères entiers puis on ajoute l'ellipse.
    """
    txt = "" if text is None else str(text)
    if max_px <= 0 or not txt:
        return ""
    if _largeur(txt) <= max_px:
        return txt
    budget = max_px - CHAR_W_MAJ            # la place de l'ellipse est réservée
    coupe = ""
    for car in txt:
        if _largeur(coupe + car) > budget:
            break
        coupe += car
    coupe = coupe.rstrip()
    return f"{coupe}{ell}" if coupe else ell


def _nice_scale(vmin, vmax, target=5):
    """Échelle Y « jolie » : bornes arrondies, pas régulier, 2 à 10 graduations.

    Renvoie (bas, haut, pas, graduations). Le zéro est inclus dès que les
    valeurs sont de signe constant, pour que les barres partent d'une base
    lisible.
    """
    vmin, vmax = _num(vmin), _num(vmax)
    if vmin is None or vmax is None:
        return 0.0, 1.0, 1.0, [0.0, 1.0]
    if vmin > 0:
        vmin = 0.0
    if vmax < 0:
        vmax = 0.0
    if vmax <= vmin:  # valeurs toutes égales (ou toutes nulles) : on ouvre la plage
        vmax = max(vmin + 1.0, vmin + abs(vmin) * 0.5)
    span = vmax - vmin
    raw = span / max(1, target)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = 10.0 * mag
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            step = mult * mag
            break
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    steps = int(round((hi - lo) / step))
    while steps > 9 and step > 0:  # trop de graduations → pas suivant
        step *= 2
        lo = math.floor(vmin / step) * step
        hi = math.ceil(vmax / step) * step
        steps = int(round((hi - lo) / step))
    ticks = [round(lo + i * step, 10) for i in range(max(0, steps) + 1)]
    if len(ticks) < 2:
        ticks = [round(lo, 10), round(lo + step, 10)]
    return lo, hi, step, ticks


def _tick_label(val, mode) -> str:
    return _fmt_duration(val, short=True) if mode == "duration" else _fmt_num(val)


def _point_value(point, metric_key=None):
    """Valeur chiffrée d'un point ; `values` (multi-métriques) prioritaire."""
    if not isinstance(point, dict):
        return None
    raw = None
    values = point.get("values")
    if metric_key and isinstance(values, dict):
        raw = values.get(metric_key)
    if raw is None:
        raw = point.get("value")
    return _num(raw)


# ---------------------------------------------------------------------------
# Normalisation défensive des données
# ---------------------------------------------------------------------------

def _normalize(data, metric_key=None):
    """Normalise l'entrée en (metric, groups, series) ; None si vide total.

    `series` est la liste alignée des séries : chacune porte `key`, `label`,
    `hidden`, `index` (position d'origine, pour la couleur de la légende),
    `values` (une valeur ou None par groupe) et `points` (le point source ou
    None par groupe). `metric_key` lit `point["values"][metric_key]` quand un
    point porte plusieurs métriques.
    """
    if not isinstance(data, dict):
        return None
    metric = data.get("metric") if isinstance(data.get("metric"), dict) else {}

    groups = []
    raw_groups = data.get("groups") if isinstance(data.get("groups"), list) else []
    for i, grp in enumerate(raw_groups):
        if isinstance(grp, dict):
            key = grp.get("key")
            key = str(key if key is not None else i)
            groups.append({"key": key, "label": str(grp.get("label") or key)})
        elif grp is not None:
            groups.append({"key": str(grp), "label": str(grp)})

    raw_series = data.get("series") if isinstance(data.get("series"), list) else []
    series_in = []
    for i, ser in enumerate(raw_series):
        if not isinstance(ser, dict):
            continue
        key = ser.get("key")
        key = str(key if key is not None else i)
        points = ser.get("points") if isinstance(ser.get("points"), list) else []
        series_in.append({
            "key": key,
            "label": str(ser.get("label") or key),
            "hidden": bool(ser.get("hidden")),
            "points": [p for p in points if isinstance(p, dict)],
        })

    if not groups:  # groupes déduits des points, dans l'ordre rencontré
        seen = {}
        for ser in series_in:
            for pt in ser["points"]:
                if pt.get("group") is not None:
                    seen.setdefault(str(pt.get("group")), str(pt.get("group")))
        groups = [{"key": k, "label": v} for k, v in seen.items()]
    if not groups or not series_in:
        return None

    alias = {}  # un point peut désigner son groupe par clé ou par libellé
    for grp in groups:
        alias[grp["key"]] = grp["key"]
        alias[grp["label"]] = grp["key"]

    series = []
    vides = []                          # séries sans aucune valeur chiffrée
    for i, ser in enumerate(series_in):
        by_group = {}
        for pt in ser["points"]:
            gk = pt.get("group")
            if gk is None:
                continue
            gk = alias.get(str(gk))
            if gk is not None and gk not in by_group:
                by_group[gk] = pt
        values = [_point_value(by_group.get(grp["key"]), metric_key) for grp in groups]
        points = [by_group.get(grp["key"]) for grp in groups]
        item = {
            "key": ser["key"], "label": ser["label"], "hidden": ser["hidden"],
            "index": i, "values": values, "points": points,
        }
        (series if any(v is not None for v in values) else vides).append(item)
    if not series:
        # Rien de chiffré : on garde les séries pour afficher « aucune donnée
        # chiffrée » plutôt qu'un trou dans la page ("" est réservé au vide total).
        series = vides
    if not series:
        return None

    metric_norm = {
        "key": str(metric.get("key") or ""),
        "label": str(metric.get("label") or metric.get("key") or "Valeur"),
        "unit": str(metric.get("unit") or ""),
        "better": str(metric.get("better") or "").lower(),
    }
    return metric_norm, groups, series


# ---------------------------------------------------------------------------
# Fragments SVG
# ---------------------------------------------------------------------------

def _svg_open(width, height, aria) -> str:
    """Ouvre un SVG fluide, autonome, héritant de la police de la page."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(width)} {int(height)}" '
        f'width="100%" height="{int(height)}" preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="{_esc(aria)}" '
        f'style="display:block;width:100%;max-width:{int(width)}px;height:auto;'
        f'font-family:inherit">'
    )


def _message_svg(width, height, message) -> str:
    """SVG minimal « aucune donnée » — jamais d'exception."""
    width, height = int(width), max(_H_MESSAGE, min(int(height), 200))
    return (
        _svg_open(width, height, message)
        + f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
          f'font-size="13" fill="{C_DIM}" font-family="inherit">{_esc(message)}</text>'
        + "</svg>"
    )


def _tooltip(serie, groupe, value, unit, duration_s, date, status) -> str:
    """Info-bulle native : 6 lignes, sans une ligne de JavaScript."""
    lignes = [
        f"Série : {serie}",
        f"Groupe : {groupe}",
        f"Valeur : {_fmt_value(value, unit)}",
        f"Durée : {_fmt_duration(duration_s)}",
        f"Date : {_fmt_date(date)}",
        f"Statut : {_fmt_status(status)}",
    ]
    return "<title>" + _esc("\n".join(lignes)) + "</title>"


def _titre_axes(metric, width, mode) -> str:
    """Titre de l'axe Y (avec unité) + rappel du sens de lecture."""
    label, unit = metric["label"], metric["unit"]
    if mode == "duration":
        titre = f"{label} ({unit or 's'})"
    else:
        titre = f"{label} ({unit})" if unit else label
    hint = {"low": "plus bas = mieux", "high": "plus haut = mieux"}.get(metric["better"], "")
    out = (
        f'<text x="12" y="18" font-size="{FS_TITLE}" font-weight="600" fill="{C_FG}" '
        f'font-family="inherit">{_esc(titre)}</text>'
    )
    if hint:
        out += (
            f'<text x="{int(width) - 12}" y="18" text-anchor="end" font-size="{FS_LABEL}" '
            f'fill="{C_DIM}" font-family="inherit">{_esc(hint)}</text>'
        )
    return out


def _y_axis(lo, hi, ticks, left, top, plot_w, plot_h, mode) -> str:
    """Grille horizontale, graduations « jolies » et axe du zéro."""
    out = []
    span = (hi - lo) or 1.0
    for val in ticks:
        y = top + plot_h - (val - lo) / span * plot_h
        out.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="{C_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 8:.1f}" y="{y + 3.6:.1f}" text-anchor="end" '
            f'font-size="{FS_LABEL}" fill="{C_DIM}" font-family="inherit">'
            f'{_esc(_tick_label(val, mode))}</text>'
        )
    if lo < 0 < hi:  # des valeurs négatives : l'axe du zéro est marqué
        y0 = top + plot_h - (0.0 - lo) / span * plot_h
        out.append(
            f'<line x1="{left:.1f}" y1="{y0:.1f}" x2="{left + plot_w:.1f}" y2="{y0:.1f}" '
            f'stroke="{C_ZERO}" stroke-width="1.5"/>'
        )
    out.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_h:.1f}" '
        f'stroke="{C_ZERO}" stroke-width="1"/>'
    )
    return "".join(out)


def _bar(x, y_top, y_zero, bw, color, hidden, serie_key, entry_id, tips) -> str:
    """Une barre : un rectangle plein (pas de dégradé) + son info-bulle `<title>`."""
    y = min(y_top, y_zero)
    h = max(1.0, abs(y_zero - y_top))
    data_entry = f' data-entry="{_esc(entry_id)}"' if entry_id else ""
    return (
        f'<rect class="chart-bar serie" data-serie="{_esc(serie_key)}"{data_entry} '
        f'x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" '
        f'fill="{color}" fill-opacity="{0.35 if hidden else 0.92}">{tips}</rect>'
    )


# ---------------------------------------------------------------------------
# Cœur du rendu
# ---------------------------------------------------------------------------

def _render(data, width, height, *, mode="metric", metric_key=None, simple=False):
    """Barres groupées (`simple=False`) ou barres simples, en SVG inline."""
    width = max(240, int(width))
    height = max(140, int(height))
    norm = _normalize(data, metric_key)
    if norm is None:
        return ""                       # rien d'exploitable : la page n'affiche rien
    metric, groups, series = norm

    valeurs = [v for ser in series for v in ser["values"] if v is not None]
    if not valeurs:                     # des tests existent mais rien de chiffré
        return _message_svg(width, height, "aucune donnée chiffrée")
    lo, hi, _step, ticks = _nice_scale(min(valeurs), max(valeurs))
    span = (hi - lo) or 1.0
    tick_txt = [_tick_label(t, mode) for t in ticks]
    left = min(110, max(38, 16 + int(max(_largeur(t) for t in tick_txt))))
    plot_w = max(80.0, width - left - _MARGIN_RIGHT)
    plot_h = max(50.0, height - _MARGIN_TOP - _MARGIN_BOTTOM)
    top = float(_MARGIN_TOP)
    base = top + plot_h

    def y_of(val):
        return top + plot_h - (val - lo) / span * plot_h

    y_zero = y_of(0.0)
    out = [_svg_open(width, height, f"{metric['label']} — graphique")]

    # --- disposition ---
    # Bornes de lisibilité : au-delà, on dessine moins de séries (les groupes de
    # l'axe X restent tous présents) pour que les barres ne deviennent pas des
    # traits illisibles. La légende, elle, liste toujours toutes les séries.
    dessinees = series
    if len(dessinees) > _MAX_SERIES:
        dessinees = dessinees[:_MAX_SERIES]
    quota = max(1, _MAX_BARS // max(1, len(groups)))
    if len(dessinees) > quota:
        dessinees = dessinees[:quota]

    placements = []                     # (index, série, groupe, valeur, point, x, largeur)
    label_step = 1
    if simple:                          # une barre par (série, groupe)
        slots = [(j, ser, gi) for j, ser in enumerate(dessinees)
                 for gi in range(len(groups))]
        n_slots = max(1, len(slots))
        sw = plot_w / n_slots
        for i, (j, ser, gi) in enumerate(slots):
            placements.append((j, ser, groups[gi], ser["values"][gi], ser["points"][gi],
                               left + i * sw, max(1.0, sw - 3.0)))
        if sw < 26:                     # libellés un sur k pour rester lisibles
            label_step = int(math.ceil(26.0 / max(1.0, sw)))
    else:                               # plusieurs séries par groupe
        n_grp = len(groups)
        gw = plot_w / max(1, n_grp)
        band = max(1.0, gw * 0.74)
        bw = band / len(dessinees)
        gap = 1.5 if bw > 4 else 0.0
        for gi, grp in enumerate(groups):
            for j, ser in enumerate(dessinees):
                x = left + gi * gw + (gw - band) / 2.0 + j * bw
                placements.append((j, ser, grp, ser["values"][gi], ser["points"][gi],
                                   x, max(1.0, bw - gap)))

    out.append(_titre_axes(metric, width, mode))
    out.append(_y_axis(lo, hi, ticks, left, top, plot_w, plot_h, mode))

    # --- barres + valeur au-dessus quand la place le permet ---
    for _j, ser, grp, val, pt, x, bw in placements:
        if val is None:
            continue
        color = PALETTE[ser["index"] % len(PALETTE)]
        dur = _num_pur(pt.get("duration_s")) if pt else None
        tips = _tooltip(ser["label"], grp["label"], val, metric["unit"], dur,
                        pt.get("date") if pt else None,
                        pt.get("status") if pt else None)
        y_val = y_of(val)
        out.append(_bar(x, y_val, y_zero, bw, color, ser["hidden"], ser["key"],
                        pt.get("entry_id") if pt else None, tips))
        txt = _fmt_duration(val, short=True) if mode == "duration" else _fmt_num(val)
        if bw >= 20 and _largeur(txt) <= bw - 3:  # valeur affichée si la place le permet
            yt = y_val - 5 if val >= 0 else y_val + 13
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{yt:.1f}" text-anchor="middle" '
                f'font-size="{FS_VALUE}" fill="{C_DIM}" font-family="inherit">{_esc(txt)}</text>'
            )

    # --- libellés sous l'axe X (tronqués au caractère, jamais superposés) ---
    if simple:
        for i, (_j, _ser, grp, _val, _pt, x, bw) in enumerate(placements):
            if i % label_step:
                continue
            lab = _trunc_px(grp["label"], bw * label_step - 4)
            if not lab:
                continue
            tip = f'<title>{_esc(grp["label"])}</title>' if lab != grp["label"] else ""
            out.append(
                f'<text x="{x + bw * label_step / 2:.1f}" y="{base + 16:.1f}" '
                f'text-anchor="middle" font-size="{FS_LABEL}" fill="{C_DIM}" '
                f'font-family="inherit">{_esc(lab)}{tip}</text>'
            )
    else:
        gw = plot_w / max(1, len(groups))
        for gi, grp in enumerate(groups):
            lab = _trunc_px(grp["label"], gw - 4)
            if not lab:
                continue
            tip = f'<title>{_esc(grp["label"])}</title>' if lab != grp["label"] else ""
            out.append(
                f'<text x="{left + gi * gw + gw / 2:.1f}" y="{base + 16:.1f}" '
                f'text-anchor="middle" font-size="{FS_LABEL}" fill="{C_DIM}" '
                f'font-family="inherit">{_esc(lab)}{tip}</text>'
            )
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# API publique (contrat §6)
# ---------------------------------------------------------------------------

def grouped_bars(data: dict, *, width=980, height=420) -> str:
    """Barres groupées : plusieurs séries par groupe, tooltips `<title>` natifs."""
    try:
        return _render(data, width, height)
    except Exception:  # robustesse : jamais d'exception remontée à la page
        return _message_svg(max(240, int(width)), max(140, int(height)),
                            "graphique indisponible")


def bars(data: dict, *, metric_key=None, width=980, height=320) -> str:
    """Barres simples : une métrique, une barre par (série, groupe).

    `metric_key` sert à lire une métrique nommée dans `point["values"]` quand
    le point en porte plusieurs ; sinon `point["value"]` est utilisé.
    """
    try:
        return _render(data, width, height, simple=True, metric_key=metric_key)
    except Exception:
        return _message_svg(max(240, int(width)), max(140, int(height)),
                            "graphique indisponible")


def duration_chart(data: dict) -> str:
    """Le temps passé : barres par (série, groupe), en secondes, format FR."""
    try:
        norm = _normalize(data)
        if norm is None:
            return ""
        _metric, groups, series = norm
        series_dur = []
        for ser in series:
            points = []
            for pt in ser["points"]:
                dur = _num_pur(pt.get("duration_s")) if pt else None
                if dur is None:
                    continue
                points.append(dict(pt, value=dur))
            series_dur.append({"key": ser["key"], "label": ser["label"],
                               "hidden": ser["hidden"], "points": points})
        if not any(s["points"] for s in series_dur):
            return _message_svg(DEFAULT_W, 380, "aucune durée enregistrée")
        data_dur = {
            "metric": {"key": "duree_s", "label": "Temps passé par test",
                       "unit": "s", "better": "low"},
            "groups": groups,
            "series": series_dur,
        }
        return _render(data_dur, DEFAULT_W, 380, mode="duration")
    except Exception:
        return _message_svg(DEFAULT_W, 380, "graphique indisponible")


def legend(data: dict) -> str:
    """Légende HTML : pastille de couleur + libellé + nombre de points.

    Les cases à cocher portent `data-serie` : la page peut filtrer en JS, mais
    sans JS la légende reste lisible (couleur, libellé, compteur, état
    « masquée »). Le compteur ne compte que les points **chiffrés**.
    """
    try:
        if not isinstance(data, dict):
            return ""
        raw_series = data.get("series")
        if not isinstance(raw_series, list) or not raw_series:
            return ""
        parts = [
            '<div class="chart-legend" role="list" style="display:flex;flex-wrap:wrap;'
            'gap:6px 16px;align-items:center;font-family:inherit;font-size:13px;'
            'color:#e6e8ee;margin:8px 0 2px">'
        ]
        n_items = 0
        for i, ser in enumerate(raw_series):
            if not isinstance(ser, dict):
                continue
            key = ser.get("key")
            key = str(key if key is not None else i)
            label = str(ser.get("label") or key)
            hidden = bool(ser.get("hidden"))
            raw_points = ser.get("points")
            points = raw_points if isinstance(raw_points, list) else []
            n = sum(1 for pt in points if _point_value(pt) is not None)
            color = PALETTE[i % len(PALETTE)]
            suffixe = " — masquée" if hidden else ""
            pluriel = "s" if n != 1 else ""
            parts.append(
                f'<label class="chart-legend-item" role="listitem" data-serie="{_esc(key)}" '
                f'title="{_esc(label + suffixe)}" style="display:inline-flex;'
                f'align-items:center;gap:6px;cursor:pointer;opacity:{0.5 if hidden else 1}">'
                f'<input type="checkbox" class="chart-toggle" data-serie="{_esc(key)}"'
                f'{" checked" if not hidden else ""} '
                f'style="margin:0;accent-color:{color};cursor:pointer">'
                f'<span class="dot" aria-hidden="true" style="width:11px;height:11px;'
                f'border-radius:3px;background:{color};display:inline-block"></span>'
                f'<span class="lab">{_esc(label)}{_esc(suffixe)}</span>'
                f'<span class="n" style="color:#98a0b3">{n} point{pluriel}</span>'
                f'</label>'
            )
            n_items += 1
        if not n_items:
            return ""
        parts.append("</div>")
        return "".join(parts)
    except Exception:
        return ""
