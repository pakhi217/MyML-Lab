"""
Top navigation bar: logo, page title, search box, theme toggle,
notification icon and profile avatar.
"""

import streamlit as st
from components.theme import toggle_theme


def render_navbar(page_title, page_sub=""):
    left, right = st.columns([5, 1.4])

    with left:
        st.markdown(
            f"""
            <div class="topnav">
                <div>
                    <div class="topnav-title">{page_title}</div>
                    <div class="topnav-sub">{page_sub}</div>
                </div>
                <div class="topnav-search">🔍&nbsp;&nbsp;Search datasets, models, metrics...</div>
                <div class="topnav-icons">
                    <div class="icon-pill">🔔</div>
                    <div class="avatar-pill">MB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        mode = st.session_state.get("theme_mode", "dark")
        label = "🌙  Dark" if mode == "dark" else "☀️  Light"
        st.markdown("<div style='margin-top:0.05rem;'></div>", unsafe_allow_html=True)
        if st.button(label, key="theme_toggle_btn", use_container_width=True):
            toggle_theme()
            st.rerun()
