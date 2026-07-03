import streamlit as st
from components.cards import metric_row, section_title, section_caption, info_card, empty_state
from components.theme import get_theme


def render():
    theme, _ = get_theme()
    df = st.session_state.get("df")

    st.markdown(
        """
        <div class="card" style="background:linear-gradient(135deg, var(--card), var(--bg-secondary)); margin-bottom:1.4rem;">
            <div style="font-size:1.3rem; font-weight:800;">Welcome back 👋</div>
            <div style="color:var(--text-secondary); margin-top:0.3rem; font-size:0.9rem;">
                MyML Lab is a machine learning workbench built entirely from scratch with NumPy —
                upload a dataset, train an algorithm, and explore the results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Overview", "📊")
    rows = len(df) if df is not None else 0
    cols = len(df.columns) if df is not None else 0
    trained = st.session_state.get("results") is not None
    n_algos_available = 7

    metric_row([
        {"label": "Dataset Rows", "value": f"{rows:,}", "icon": "🧾"},
        {"label": "Dataset Columns", "value": f"{cols}", "icon": "📐"},
        {"label": "Algorithms Available", "value": f"{n_algos_available}", "icon": "🧠"},
        {"label": "Models Trained", "value": "1" if trained else "0", "icon": "⚙️"},
    ])

    section_title("Datasets", "📂")
    if df is not None:
        info_card(
            st.session_state.get("filename", "dataset.csv"),
            f"{rows} rows · {cols} columns · uploaded and ready for training",
            badge="Active", badge_type="green",
        )
    else:
        empty_state("No dataset uploaded yet. Head to the Dataset page to get started.", "📂")

    section_title("Recent Model", "🧠")
    results = st.session_state.get("results")
    if results:
        info_card(
            f"{results.get('algorithm', 'Model')} · {results.get('type', '').title()}",
            "Trained successfully. View full metrics and charts on the Metrics page.",
            badge="Trained", badge_type="pink",
        )
    else:
        empty_state("No models trained yet. Choose an algorithm page to train your first model.", "🧠")

    section_title("Quick Actions", "⚡")
    section_caption("Jump straight into the workflow.")
    c1, c2, c3, c4 = st.columns(4)
    actions = [
        (c1, "📂 Upload Dataset", "Dataset"),
        (c2, "📈 Train Regression", "Regression"),
        (c3, "📊 Train Classification", "Classification"),
        (c4, "⚖️ Compare Models", "Compare"),
    ]
    for col, label, target in actions:
        with col:
            if st.button(label, use_container_width=True, key=f"quick_{target}"):
                st.session_state.active_page = target
                st.rerun()
