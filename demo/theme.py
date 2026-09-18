"""The demo's visual identity, applied by every page.

The palette and type are lifted from the landing page (landing/src/components/Hero.tsx) so the
marketing page and the app read as one product: Plus Jakarta Sans over Inter, emerald on a deep
ink ground with a teal bias. `.streamlit/config.toml` sets the colours Streamlit needs at boot;
this module adds the typography and the component styling that config cannot reach.

Every page opens the same way, with `page_header`: an uppercase eyebrow naming the phase, the
title, one line saying what the page does, and an emerald hairline. That repetition is the
structure — a reader always knows where they are and which part of the system they are looking
at, without a nav bar restating it.
"""

from __future__ import annotations

import streamlit as st

INK = "#0B1014"
SURFACE = "#121A20"
EMERALD = "#10B981"

_FONTS = (
    "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700"
    "&family=Inter:wght@400;500;600&display=swap"
)

_CSS = f"""
<style>
@import url('{_FONTS}');

:root {{
  --ink: {INK};
  --surface: {SURFACE};
  --emerald: {EMERALD};
  --line: rgba(255,255,255,0.09);
  --muted: rgba(230,237,243,0.62);
}}

html, body, [class*="st-"], .stMarkdown, p, li, label, input, textarea, button {{
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}

/* Display face for every heading, set tight and large — the one loud move on the page. */
h1, h2, h3, h4,
[data-testid="stMetricValue"] {{
  font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif !important;
  letter-spacing: -0.03em;
  font-weight: 700;
}}
h1 {{ font-size: 2.9rem; line-height: 1.04; margin-bottom: 0.15rem; }}
h2 {{ font-size: 1.5rem; letter-spacing: -0.02em; margin-top: 2.2rem; }}
h3 {{ font-size: 1.15rem; letter-spacing: -0.015em; }}

/* Figures line up in columns wherever they are compared. */
[data-testid="stMetricValue"], .dbm-price, table, code {{
  font-variant-numeric: tabular-nums;
}}

/* The page opener: eyebrow, title, one line, hairline. */
.dbm-eyebrow {{
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--emerald);
  margin-bottom: 0.35rem;
}}
.dbm-lede {{
  font-size: 1.02rem;
  color: var(--muted);
  max-width: 62ch;
  margin: 0.35rem 0 0.9rem 0;
}}
.dbm-rule {{
  height: 2px;
  border: 0;
  margin: 0 0 1.6rem 0;
  background: linear-gradient(90deg, var(--emerald), rgba(16,185,129,0.15) 38%, transparent 72%);
}}

/* Cards: one hairline border and a lift on hover. No stacked shadows. */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 14px;
  border-color: var(--line) !important;
  background: var(--surface);
  transition: border-color 140ms ease, transform 140ms ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
  border-color: rgba(16,185,129,0.38) !important;
  transform: translateY(-1px);
}}

/* Chips: Streamlit's colour-background markdown, made into real pills. */
[data-testid="stMarkdownContainer"] span[style*="background-color"] {{
  border-radius: 999px !important;
  padding: 0.16rem 0.62rem !important;
  font-size: 0.8rem;
  font-weight: 500;
  line-height: 1.5;
}}

/* Buttons read as controls, and the primary action is unmistakable. */
.stButton > button {{
  border-radius: 999px;
  border: 1px solid var(--line);
  font-weight: 600;
  transition: border-color 140ms ease, background 140ms ease;
}}
.stButton > button:hover {{
  border-color: var(--emerald);
  color: #fff;
}}

/* Sidebar: its own darker ground, with the emerald rule repeated at the top. */
[data-testid="stSidebar"] {{
  background: #080C10;
  border-right: 1px solid var(--line);
}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ margin-top: 0.6rem; }}

/* Tabs: an emerald underline on the live one instead of Streamlit's default red. */
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--emerald); }}
.stTabs [aria-selected="true"] {{ color: #fff !important; }}

/* Keyboard focus stays visible, on the accent. */
:focus-visible {{ outline: 2px solid var(--emerald) !important; outline-offset: 2px; }}

/* The one big number on a card (a price, an estimate). */
.dbm-price {{
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.1;
}}

@media (prefers-reduced-motion: reduce) {{
  [data-testid="stVerticalBlockBorderWrapper"] {{ transition: none; }}
  [data-testid="stVerticalBlockBorderWrapper"]:hover {{ transform: none; }}
}}
</style>
"""


def apply() -> None:
    """Inject the fonts and component styling. Call once, at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, lede: str = "") -> None:
    """The opener every page shares: eyebrow, title, one line, emerald hairline."""
    st.markdown(f'<div class="dbm-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<p class="dbm-lede">{lede}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="dbm-rule" />', unsafe_allow_html=True)
