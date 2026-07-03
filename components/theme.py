"""
Theme engine for MyML Lab.

Holds the two color palettes (dark / light) and generates a single
large CSS block that is injected once per render. Streamlit widgets
are re-skinned via attribute selectors so the whole app (sliders,
buttons, tabs, expanders, file uploader, dataframes...) shares one
consistent premium design language.
"""

import streamlit as st

THEMES = {
    "dark": {
        "bg": "#000000",
        "bg_secondary": "#0A0A0A",
        "card": "#141414",
        "card_hover": "#1E1E1E",
        "accent": "#E91760",
        "accent_soft": "rgba(233,23,96,0.14)",
        "secondary_accent": "#F5A3C1",
        "text": "#FFFFFF",
        "text_secondary": "#A3A3A3",
        "border": "rgba(233,23,96,0.18)",
        "shadow": "0 8px 30px rgba(0,0,0,0.55)",
        "glow": "0 0 18px rgba(233,23,96,0.4)",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "chart_bg": "#0A0A0A",
        "chart_grid": "rgba(163,163,163,0.12)",
    },
    "light": {
        "bg": "#FFF5F8",
        "bg_secondary": "#FFFFFF",
        "card": "#FFFFFF",
        "card_hover": "#FDECF2",
        "accent": "#E91760",
        "accent_soft": "rgba(233,23,96,0.08)",
        "secondary_accent": "#F5A3C1",
        "text": "#111111",
        "text_secondary": "#6B6B6B",
        "border": "#F6D6E2",
        "shadow": "0 8px 24px rgba(17,17,17,0.08)",
        "glow": "0 0 16px rgba(233,23,96,0.25)",
        "success": "#059669",
        "warning": "#D97706",
        "danger": "#DC2626",
        "chart_bg": "#FFFFFF",
        "chart_grid": "rgba(17,24,39,0.06)",
    },
}


def get_theme():
    """Returns the active theme dict, defaulting to dark."""
    mode = st.session_state.get("theme_mode", "dark")
    return THEMES[mode], mode


def toggle_theme():
    current = st.session_state.get("theme_mode", "dark")
    st.session_state["theme_mode"] = "light" if current == "dark" else "dark"


def inject_global_css():
    t, mode = get_theme()

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg: {t['bg']};
        --bg-secondary: {t['bg_secondary']};
        --card: {t['card']};
        --card-hover: {t['card_hover']};
        --accent: {t['accent']};
        --accent-soft: {t['accent_soft']};
        --secondary-accent: {t['secondary_accent']};
        --text: {t['text']};
        --text-secondary: {t['text_secondary']};
        --border: {t['border']};
        --shadow: {t['shadow']};
        --glow: {t['glow']};
        --success: {t['success']};
        --warning: {t['warning']};
        --danger: {t['danger']};
    }}

    * {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: background-color 250ms ease, color 250ms ease, border-color 250ms ease, box-shadow 250ms ease;
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: var(--bg) !important;
        color: var(--text) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
        height: 0px;
    }}

    #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}

    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }}

    /* ---------------- SIDEBAR ---------------- */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem;
    }}

    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.4rem 0.6rem 1.4rem 0.6rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }}
    .sidebar-brand .logo-badge {{
        width: 34px; height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), var(--secondary-accent));
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: white; font-size: 15px;
        box-shadow: var(--glow);
    }}
    .sidebar-brand .brand-text {{
        font-weight: 800; font-size: 1.02rem; color: var(--text); line-height: 1.1;
    }}
    .sidebar-brand .brand-sub {{
        font-size: 0.68rem; color: var(--text-secondary); letter-spacing: 0.04em;
    }}

    /* Nav buttons rendered via st.button, restyled */
    section[data-testid="stSidebar"] div.stButton > button {{
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 2px;
        box-shadow: none;
    }}
    section[data-testid="stSidebar"] div.stButton > button:hover {{
        background: var(--accent-soft);
        color: var(--text);
        border-color: var(--border);
        transform: translateX(2px);
    }}
    section[data-testid="stSidebar"] div.stButton > button:focus {{
        outline: none;
        box-shadow: none;
    }}
    .nav-active > div.stButton > button {{
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
        box-shadow: inset 3px 0 0 var(--accent), var(--glow);
        font-weight: 700;
    }}

    /* ---------------- TOP NAVBAR ---------------- */
    .topnav {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 1.4rem;
        box-shadow: var(--shadow);
        animation: fadeIn 0.4s ease;
    }}
    .topnav-title {{ font-weight: 700; font-size: 1.0rem; color: var(--text); }}
    .topnav-sub {{ font-size: 0.72rem; color: var(--text-secondary); }}
    .topnav-search {{
        flex: 1;
        max-width: 320px;
        margin: 0 1.2rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
        color: var(--text-secondary);
        font-size: 0.82rem;
    }}
    .topnav-icons {{ display: flex; align-items: center; gap: 0.7rem; }}
    .icon-pill {{
        width: 36px; height: 36px;
        border-radius: 10px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 15px;
        cursor: pointer;
    }}
    .icon-pill:hover {{ box-shadow: var(--glow); border-color: var(--accent); }}
    .avatar-pill {{
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, var(--secondary-accent), var(--accent));
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.8rem;
    }}

    /* ---------------- CARDS ---------------- */
    .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: var(--shadow);
        animation: fadeIn 0.45s ease;
    }}
    .card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow), var(--glow); }}

    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: var(--shadow);
        animation: fadeIn 0.45s ease;
    }}
    .metric-card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow), var(--glow); }}
    .metric-label {{
        font-size: 0.74rem; color: var(--text-secondary);
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 1.65rem; font-weight: 800; color: var(--text);
        font-family: 'JetBrains Mono', monospace;
    }}
    .metric-delta-up {{ color: var(--success); font-size: 0.78rem; font-weight: 600; }}
    .metric-delta-down {{ color: var(--danger); font-size: 0.78rem; font-weight: 600; }}

    .badge {{
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.03em;
    }}
    .badge-pink {{ background: var(--accent-soft); color: var(--accent); }}
    .badge-purple {{ background: rgba(139,92,246,0.14); color: var(--secondary-accent); }}
    .badge-green {{ background: rgba(52,211,153,0.14); color: var(--success); }}

    .section-title {{
        font-size: 1.05rem; font-weight: 800; color: var(--text);
        margin: 1.6rem 0 0.8rem 0; display: flex; align-items: center; gap: 8px;
    }}
    .section-caption {{ color: var(--text-secondary); font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem; }}

    .formula-box {{
        background: var(--bg-secondary);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-size: 0.95rem;
        margin: 0.6rem 0 1rem 0;
    }}

    /* ---------------- BUTTONS ---------------- */
    div.stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--secondary-accent));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.3rem;
        font-weight: 700;
        font-size: 0.88rem;
        box-shadow: 0 4px 14px rgba(255,46,136,0.25);
    }}
    div.stButton > button:hover, .stDownloadButton > button:hover {{
        box-shadow: var(--glow);
        transform: translateY(-1px);
    }}
    div.stButton > button:active {{ transform: translateY(0px); }}

    /* ---------------- INPUTS / WIDGETS ---------------- */
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--bg-secondary) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: 14px !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--accent) !important; }}

    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stTextInput input, .stNumberInput input {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }}

    .stSlider [data-baseweb="slider"] > div > div {{ background: var(--accent) !important; }}
    .stSlider [role="slider"] {{
        background: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-soft) !important;
    }}

    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--accent), var(--secondary-accent)) !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--text-secondary);
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }}

    [data-testid="stExpander"] {{
        background: var(--card);
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    div[data-baseweb="notification"] {{
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    /* ---------------- ANIMATIONS ---------------- */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 0 0 rgba(255,46,136,0); }}
        50% {{ box-shadow: 0 0 16px rgba(255,46,136,0.4); }}
    }}
    .skeleton {{
        background: linear-gradient(90deg, var(--card) 25%, var(--card-hover) 50%, var(--card) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.4s infinite;
        border-radius: 10px;
    }}
    @keyframes shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return t, mode
