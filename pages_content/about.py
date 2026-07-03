import streamlit as st
from components.cards import section_title, section_caption, info_card


def render():
    section_title("About MyML Lab", "ℹ️")
    section_caption("A machine learning library and dashboard built entirely from first principles.")

    st.markdown(
        """
        <div class="card">
            <div style="font-weight:800; font-size:1.05rem; margin-bottom:0.4rem;">Why this project exists</div>
            <div style="color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                MyML Lab implements classic machine learning algorithms — regression, classification,
                clustering, and dimensionality reduction — using only NumPy for the math. No scikit-learn
                is used inside the algorithms themselves; it only appears on the Compare page as a
                reference benchmark. The goal is to understand these algorithms deeply enough to
                re-derive them, not just call a library function.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Implemented Algorithms", "🧠")
    c1, c2 = st.columns(2)
    with c1:
        info_card("Linear Regression", "Gradient descent, MSE loss, optional L2 regularization.", "Regression", "pink")
        info_card("Logistic Regression", "Sigmoid activation, binary cross-entropy loss.", "Classification", "purple")
        info_card("K-Nearest Neighbors", "Distance-based majority vote classifier.", "Classification", "purple")
    with c2:
        info_card("Decision Tree", "Gini-impurity based recursive splitting.", "Classification", "purple")
        info_card("Random Forest", "Bootstrap aggregation of decision trees.", "Classification", "purple")
        info_card("K-Means / PCA", "Centroid-based clustering and eigen-decomposition based reduction.", "Unsupervised", "green")

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Tech Stack", "🛠️")
    st.markdown(
        """
        <div class="card">
            <b>Frontend:</b> Streamlit, custom CSS, Plotly<br/>
            <b>ML Core:</b> NumPy, Pandas<br/>
            <b>Reports:</b> ReportLab (PDF), JSON, CSV export<br/>
            <b>Benchmarking:</b> scikit-learn (Compare page only)
        </div>
        """,
        unsafe_allow_html=True,
    )
