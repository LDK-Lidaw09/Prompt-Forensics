"""Look of the PSI demo application: style sheet and small HTML components.

Only presentation lives here. The forensic logic stays in the
``prompt_forensics`` package and the page layout in ``streamlit_app.py``.
Every component returns an HTML string that the app renders with
``st.html``; all text coming from data is escaped.
"""

from __future__ import annotations

import html
from typing import Iterable

# Colours shared by the style sheet and the components.
INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
ACCENT = "#2563eb"
VERDICT = {"safe": "#16a34a", "suspicious": "#f59e0b", "malicious": "#dc2626"}
UNKNOWN = "#94a3b8"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --psi-ink: #0f172a; --psi-muted: #64748b; --psi-line: #e2e8f0;
  --psi-accent: #2563eb; --psi-accent-2: #7c3aed; --psi-cyan: #06b6d4;
  --psi-card: #ffffff; --psi-bg: #f5f7fb;
  --psi-radius: 16px;
  --psi-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06);
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
code, pre, .psi-mono { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important; }

.stApp { background: var(--psi-bg); }
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1280px; }
.stAppDeployButton, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 { color: var(--psi-ink); letter-spacing: -0.02em; }
h3 { font-weight: 700; }

/* ---------- tabs as a pill navigation bar ----------
   Role selectors work with both tab implementations of Streamlit (BaseWeb up to
   about 1.5x, React Aria after). */
.stTabs [role="tablist"] {
  gap: 4px; background: var(--psi-card); padding: 6px; border-radius: 999px;
  box-shadow: var(--psi-shadow); border: 1px solid var(--psi-line);
  overflow-x: auto; scrollbar-width: none; flex-wrap: nowrap; border-bottom: 1px solid var(--psi-line);
}
.stTabs [role="tablist"]::-webkit-scrollbar { display: none; }
.stTabs [role="tab"] {
  height: 40px; padding: 0 16px; border-radius: 999px; white-space: nowrap; flex: none;
  display: flex; align-items: center; margin: 0;
  color: var(--psi-muted); font-weight: 600; transition: background .15s, color .15s; cursor: pointer;
}
.stTabs [role="tab"] p { font-weight: 600; margin: 0; }
.stTabs [role="tab"]:hover { background: #eef2ff; color: var(--psi-accent); }
.stTabs [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, var(--psi-accent), var(--psi-accent-2)) !important;
  color: #fff !important; box-shadow: 0 6px 16px rgba(37,99,235,.28);
}
.stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] span { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator { display: none !important; }
.stTabs [role="tabpanel"] { padding-top: 1.4rem; }
/* tabs nested inside a tab stay discreet */
.stTabs .stTabs [role="tablist"] { box-shadow: none; background: #f1f5f9; }

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--psi-card); border-radius: var(--psi-radius) !important;
  border-color: var(--psi-line) !important; box-shadow: var(--psi-shadow);
}
[data-testid="stMetric"] {
  background: var(--psi-card); border: 1px solid var(--psi-line); border-radius: 14px;
  padding: 14px 16px; box-shadow: var(--psi-shadow); position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, var(--psi-accent), var(--psi-accent-2));
}
[data-testid="stMetricLabel"] p { color: var(--psi-muted); font-weight: 600; font-size: .82rem;
  text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricValue"] { font-weight: 800; color: var(--psi-ink); }
[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--psi-line);
  background: var(--psi-card); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--psi-line);
  transition: transform .1s, box-shadow .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px);
  box-shadow: 0 6px 14px rgba(15,23,42,.08); }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: linear-gradient(135deg, var(--psi-accent), var(--psi-accent-2)); color: #fff; border: 0;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #0b1220; }
[data-testid="stSidebar"] * { color: #cbd5e1; }
[data-testid="stSidebar"] h2 { color: #fff; font-size: 1rem; text-transform: uppercase;
  letter-spacing: .08em; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #111a2e !important; color: #e2e8f0 !important; }

/* ---------- custom components ---------- */
.psi-hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px 30px;
  background: radial-gradient(1200px 400px at 85% -10%, rgba(124,58,237,.55), transparent 60%),
              radial-gradient(800px 300px at 0% 120%, rgba(6,182,212,.35), transparent 60%),
              linear-gradient(135deg, #0b1220 0%, #13204a 55%, #1e1b4b 100%);
  color: #e2e8f0; box-shadow: 0 20px 50px rgba(15,23,42,.25); margin-bottom: 18px;
}
.psi-hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .18;
  background-image: linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px);
  background-size: 28px 28px;
}
.psi-hero > * { position: relative; z-index: 1; }
.psi-eyebrow { font-size: .74rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
  color: #67e8f9; }
.psi-hero h1 { color: #fff; font-size: clamp(1.7rem, 4vw, 2.7rem); font-weight: 800; margin: 6px 0 6px;
  line-height: 1.1; padding: 0; }
.psi-hero h1 span { background: linear-gradient(90deg, #67e8f9, #a5b4fc, #f0abfc);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.psi-hero p { color: #cbd5e1; font-size: 1.02rem; max-width: 760px; margin: 0; }
.psi-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.psi-chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); color: #e2e8f0;
  font-size: .8rem; font-weight: 600; backdrop-filter: blur(4px); }
.psi-chip.warn { background: rgba(245,158,11,.16); border-color: rgba(245,158,11,.45); color: #fde68a; }
.psi-hero-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-top: 22px; }
.psi-hero-stat { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
  border-radius: 14px; padding: 12px 14px; }
.psi-hero-stat b { display: block; font-size: 1.45rem; color: #fff; font-weight: 800; }
.psi-hero-stat small { color: #94a3b8; font-size: .78rem; }

.psi-section { display: flex; align-items: flex-start; gap: 12px; margin: 6px 0 4px; }
.psi-section .ico { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
  font-size: 1.2rem; color: #fff; flex: none;
  background: linear-gradient(135deg, var(--psi-accent), var(--psi-accent-2));
  box-shadow: 0 6px 16px rgba(37,99,235,.25); }
.psi-section h3 { margin: 0; padding: 0; font-size: 1.45rem; }
.psi-section p { margin: 2px 0 0; color: var(--psi-muted); font-size: .92rem; }

.psi-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.psi-card { background: var(--psi-card); border: 1px solid var(--psi-line); border-radius: var(--psi-radius);
  padding: 18px 18px 16px; box-shadow: var(--psi-shadow); transition: transform .15s, box-shadow .15s; }
.psi-card:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(15,23,42,.10); }
.psi-card .num { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  font-weight: 800; color: #fff; margin-bottom: 10px; }
.psi-card h4 { margin: 0 0 6px; font-size: 1.08rem; }
.psi-card .q { color: var(--psi-ink); font-weight: 500; margin: 0 0 8px; }
.psi-card .d { color: var(--psi-muted); font-size: .86rem; margin: 0; }
.psi-card .tabs { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.psi-tag { font-size: .72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px;
  background: #eef2ff; color: #4338ca; }

.psi-stat { background: var(--psi-card); border: 1px solid var(--psi-line); border-radius: 14px;
  padding: 16px; box-shadow: var(--psi-shadow); position: relative; overflow: hidden; }
.psi-stat::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px;
  background: var(--c, var(--psi-accent)); }
.psi-stat .v { font-size: 1.9rem; font-weight: 800; color: var(--psi-ink); line-height: 1.1; }
.psi-stat .l { color: var(--psi-muted); font-size: .82rem; font-weight: 600; margin-top: 4px; }
.psi-stat .s { color: var(--psi-muted); font-size: .76rem; margin-top: 6px; }

.psi-flow { display: flex; align-items: stretch; gap: 10px; flex-wrap: wrap; margin: 6px 0 2px; }
.psi-flow .step { flex: 1 1 160px; background: var(--psi-card); border: 1px solid var(--psi-line);
  border-radius: 14px; padding: 12px 14px; box-shadow: var(--psi-shadow); }
.psi-flow .step b { display: block; color: var(--psi-ink); }
.psi-flow .step span { color: var(--psi-muted); font-size: .82rem; }
.psi-flow .arrow { align-self: center; color: #a5b4fc; font-size: 1.4rem; font-weight: 800; }

.psi-tour { counter-reset: tour; display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.psi-tour div { counter-increment: tour; background: var(--psi-card); border: 1px dashed #c7d2fe;
  border-radius: 14px; padding: 12px 14px 12px 50px; position: relative; color: var(--psi-ink);
  font-size: .9rem; }
.psi-tour div::before { content: counter(tour); position: absolute; left: 14px; top: 12px; width: 26px;
  height: 26px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; color: #fff;
  background: linear-gradient(135deg, var(--psi-accent), var(--psi-accent-2)); font-size: .8rem; }

.psi-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .74rem;
  font-weight: 700; color: #fff; text-transform: uppercase; letter-spacing: .05em; }

.psi-prompt { background: var(--psi-card); border: 1px solid var(--psi-line); border-radius: var(--psi-radius);
  box-shadow: var(--psi-shadow); padding: 18px 20px; border-left: 5px solid var(--c, var(--psi-accent)); }
.psi-prompt .meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.psi-prompt .text { color: var(--psi-ink); white-space: pre-wrap; line-height: 1.55; font-size: .95rem; }

.psi-strip { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 4px; max-width: 640px; }
.psi-strip span { aspect-ratio: 1; border-radius: 4px; transition: transform .1s; }
.psi-strip span:hover { transform: scale(1.35); box-shadow: 0 0 0 2px #fff, 0 0 0 4px currentColor; }
.psi-legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--psi-muted); font-size: .82rem;
  margin-top: 10px; }
.psi-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; }

.psi-callout { border-radius: 14px; padding: 14px 16px; display: flex; gap: 12px; align-items: flex-start;
  border: 1px solid; font-size: .93rem; }
.psi-callout .i { font-size: 1.3rem; line-height: 1; }
.psi-callout.red { background: #fef2f2; border-color: #fecaca; color: #7f1d1d; }
.psi-callout.amber { background: #fffbeb; border-color: #fde68a; color: #78350f; }
.psi-callout.blue { background: #eff6ff; border-color: #bfdbfe; color: #1e3a8a; }
.psi-callout.green { background: #f0fdf4; border-color: #bbf7d0; color: #14532d; }

.psi-chain { display: flex; gap: 8px; flex-wrap: nowrap; align-items: stretch; margin: 10px 0;
  overflow-x: auto; padding: 4px 2px 10px; }
.psi-block { background: var(--psi-card); border: 1px solid var(--psi-line); border-radius: 14px;
  padding: 10px 12px; flex: 0 0 168px; font-size: .78rem;
  box-shadow: var(--psi-shadow); border-top: 4px solid var(--psi-accent); }
.psi-block.bad { border-top-color: #dc2626; background: #fff5f5; border-color: #fecaca; }
.psi-block .h { font-weight: 800; color: var(--psi-ink); display: flex; justify-content: space-between; gap: 6px; }
.psi-block.bad .h { color: #b91c1c; }
.psi-block .p { color: #475569; margin: 6px 0; height: 3.3em; overflow: hidden; }
.psi-block .m { font-family: 'JetBrains Mono', monospace; color: #64748b; font-size: .72rem; }
.psi-link { align-self: center; color: #a5b4fc; font-size: 1.3rem; font-weight: 800; flex: none; }
.psi-link.bad { color: #dc2626; }
.psi-more { align-self: center; color: var(--psi-muted); font-size: .82rem; flex: none; white-space: nowrap; }

/* ---------- small screens ---------- */
@media (max-width: 760px) {
  .block-container { padding-left: 14px; padding-right: 14px; }
  .psi-hero { padding: 22px 18px; border-radius: 18px; }
  .psi-hero p { font-size: .92rem; }
  .psi-hero-stat b { font-size: 1.15rem; }
  .psi-strip { grid-template-columns: repeat(10, minmax(0, 1fr)); }
  .psi-flow .arrow { display: none; }
  .stTabs [role="tab"] { padding: 0 12px; height: 36px; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
}
</style>
"""


def esc(value: object) -> str:
    return html.escape(str(value))


def hero(title: str, highlight: str, subtitle: str, chips: Iterable[tuple[str, bool]] = (),
         stats: Iterable[tuple[str, str]] = ()) -> str:
    chip_html = "".join(f'<span class="psi-chip{" warn" if warn else ""}">{esc(text)}</span>'
                        for text, warn in chips)
    stat_html = "".join(f'<div class="psi-hero-stat"><b>{esc(v)}</b><small>{esc(label)}</small></div>'
                        for v, label in stats)
    return (f'<div class="psi-hero"><div class="psi-eyebrow">Prompt forensics</div>'
            f'<h1>{esc(title)} <span>{esc(highlight)}</span></h1><p>{esc(subtitle)}</p>'
            + (f'<div class="psi-chips">{chip_html}</div>' if chip_html else "")
            + (f'<div class="psi-hero-stats">{stat_html}</div>' if stat_html else "")
            + "</div>")


def section(icon: str, title: str, subtitle: str = "") -> str:
    return (f'<div class="psi-section"><div class="ico">{icon}</div><div><h3>{esc(title)}</h3>'
            + (f"<p>{esc(subtitle)}</p>" if subtitle else "") + "</div></div>")


def stat_cards(items: Iterable[tuple[str, str, str, str]]) -> str:
    """``items`` = (value, label, small print, colour)."""
    cards = "".join(
        f'<div class="psi-stat" style="--c:{colour}"><div class="v">{esc(v)}</div>'
        f'<div class="l">{esc(label)}</div>' + (f'<div class="s">{esc(small)}</div>' if small else "")
        + "</div>" for v, label, small, colour in items)
    return f'<div class="psi-grid" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr))">{cards}</div>'


def badge(text: str, colour: str) -> str:
    return f'<span class="psi-badge" style="background:{colour}">{esc(text)}</span>'


def callout(kind: str, icon: str, text: str) -> str:
    """``text`` may contain <b> tags; callers escape the data they insert."""
    return f'<div class="psi-callout {kind}"><div class="i">{icon}</div><div>{text}</div></div>'
