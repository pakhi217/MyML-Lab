"""
Reusable HTML card components rendered via st.markdown.
Keeping these as small render functions keeps every page visually consistent.
"""

import streamlit as st


def metric_card(label, value, delta=None, delta_positive=True, icon=""):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-up" if delta_positive else "metric-delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items, columns=None):
    """items: list of dicts with keys label, value, delta(optional), delta_positive(optional), icon(optional)"""
    n = len(items)
    cols = st.columns(columns or n)
    for col, item in zip(cols, items):
        with col:
            metric_card(
                item.get("label", ""),
                item.get("value", "-"),
                item.get("delta"),
                item.get("delta_positive", True),
                item.get("icon", ""),
            )


def info_card(title, description, badge=None, badge_type="pink"):
    badge_html = f'<span class="badge badge-{badge_type}">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <div style="font-weight:700; font-size:0.95rem;">{title}</div>
                {badge_html}
            </div>
            <div style="color:var(--text-secondary); font-size:0.85rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text, icon=""):
    st.markdown(f'<div class="section-title">{icon} {text}</div>', unsafe_allow_html=True)


def section_caption(text):
    st.markdown(f'<div class="section-caption">{text}</div>', unsafe_allow_html=True)


def formula_box(formula_text):
    st.markdown(f'<div class="formula-box">{formula_text}</div>', unsafe_allow_html=True)


def empty_state(message, icon="📭"):
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding:2.4rem 1rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="color:var(--text-secondary); font-weight:600;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
