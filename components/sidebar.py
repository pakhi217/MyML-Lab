"""
Custom left sidebar navigation. Built with st.button (not the native
Streamlit multipage nav) so we get full control over the neon
glow-on-active styling required by the design spec.
"""

import streamlit as st

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Dataset", "📂"),
    ("Regression", "📈"),
    ("Classification", "📊"),
    ("Clustering", "🔍"),
    ("PCA", "📉"),
    ("Metrics", "📚"),
    ("Compare", "⚖️"),
    ("Reports", "📄"),
    ("About", "ℹ️"),
]


def render_sidebar():
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Dashboard"

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo-badge">ML</div>
                <div>
                    <div class="brand-text">MyML Lab</div>
                    <div class="brand-sub">MACHINE LEARNING · FROM SCRATCH</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for name, icon in NAV_ITEMS:
            is_active = st.session_state.active_page == name
            wrapper_class = "nav-active" if is_active else ""
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}   {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.active_page = name
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.4rem; border-top:1px solid var(--border);'></div>", unsafe_allow_html=True)

        ds_loaded = st.session_state.get("df") is not None
        status = "🟢 Dataset loaded" if ds_loaded else "⚪ No dataset yet"
        st.markdown(
            f"""
            <div style="padding:0.9rem 0.6rem; margin-top:0.8rem;
                        background:var(--bg-secondary); border:1px solid var(--border);
                        border-radius:12px; font-size:0.76rem; color:var(--text-secondary);">
                {status}<br/>
                <span style="opacity:0.8;">Built with NumPy · No sklearn core</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_page
