from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Visualisations", page_icon="📊", layout="wide")

with open(Path(__file__).parent.parent / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not st.session_state.get("authenticated", False):
    st.warning("Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

from utils.loader import load_silver, load_meta
logger.info(f"Page Visualisations — user : {st.session_state.get('username','?')}")

df  = load_silver()
meta = load_meta()

st.title("Visualisations — Analyse des donnees et du modele")
st.markdown("Exploration interactive des correlations, distributions et des performances des modeles.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribution des charges",
    "Correlations IMC, Age, Charges",
    "Analyse par groupes",
    "Interpretabilite des modeles"
])

# ── Tab 1 : Distribution ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribution des frais medicaux annuels")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(df, x="charges", nbins=50, color_discrete_sequence=["#2563eb"],
                           title="Distribution des charges (USD)",
                           labels={"charges": "Frais medicaux (USD)"})
        mean_val = df["charges"].mean()
        median_val = df["charges"].median()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne : {mean_val:,.0f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="orange",
                      annotation_text=f"Mediane : {median_val:,.0f}")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La distribution est asymetrique a droite — quelques clients ont des charges tres elevees.")

    with col_r:
        fig2 = px.box(df, x="smoker", y="charges", color="smoker",
                      color_discrete_map={"yes": "#ef4444", "no": "#22c55e"},
                      title="Charges selon le statut fumeur",
                      labels={"smoker": "Fumeur", "charges": "Frais (USD)"})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        ratio = df[df["smoker"]=="yes"]["charges"].mean() / df[df["smoker"]=="no"]["charges"].mean()
        st.caption(f"Les fumeurs ont des charges {ratio:.1f}x plus elevees en moyenne.")

    # Violin par region
    fig3 = px.violin(df, x="region", y="charges", color="region", box=True,
                     title="Distribution des charges par region",
                     labels={"region": "Region", "charges": "Frais (USD)"})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2 : Correlations IMC / Age / Charges ──────────────────────────────────
with tab2:
    st.subheader("Correlations entre l'IMC, l'age et les frais medicaux")
    st.markdown("Ce dashboard repond a la specification fonctionnelle du projet : visualiser la correlation IMC, age et charges.")

    color_option = st.selectbox("Colorier les points par :", ["smoker", "sex", "region"], key="scatter_color")
    col_l, col_r = st.columns(2)

    with col_l:
        fig_bmi = px.scatter(df, x="bmi", y="charges", color=color_option,
                             title="IMC vs Frais medicaux",
                             labels={"bmi": "IMC (BMI)", "charges": "Frais (USD)"},
                             opacity=0.6, size_max=8,
                             color_discrete_map={"yes":"#ef4444","no":"#22c55e",
                                                 "male":"#3b82f6","female":"#f59e0b"})
        # Ligne de tendance manuelle
        valid = df[["bmi","charges"]].dropna()
        m, b = np.polyfit(valid["bmi"].astype(float), valid["charges"].astype(float), 1)
        x_line = [float(valid["bmi"].min()), float(valid["bmi"].max())]
        fig_bmi.add_trace(go.Scatter(x=x_line, y=[m*x+b for x in x_line],
                                     mode="lines", line=dict(color="black", dash="dash", width=2),
                                     name="Tendance", showlegend=True))
        st.plotly_chart(fig_bmi, use_container_width=True)
        corr_bmi = df["bmi"].corr(df["charges"])
        st.caption(f"Correlation IMC / Charges : **r = {corr_bmi:.3f}**")

    with col_r:
        fig_age = px.scatter(df, x="age", y="charges", color=color_option,
                             title="Age vs Frais medicaux",
                             labels={"age": "Age (ans)", "charges": "Frais (USD)"},
                             opacity=0.6,
                             color_discrete_map={"yes":"#ef4444","no":"#22c55e",
                                                 "male":"#3b82f6","female":"#f59e0b"})
        valid2 = df[["age","charges"]].dropna()
        m2, b2 = np.polyfit(valid2["age"].astype(float), valid2["charges"].astype(float), 1)
        x_line2 = [float(valid2["age"].min()), float(valid2["age"].max())]
        fig_age.add_trace(go.Scatter(x=x_line2, y=[m2*x+b2 for x in x_line2],
                                     mode="lines", line=dict(color="black", dash="dash", width=2),
                                     name="Tendance", showlegend=True))
        st.plotly_chart(fig_age, use_container_width=True)
        corr_age = df["age"].corr(df["charges"])
        st.caption(f"Correlation Age / Charges : **r = {corr_age:.3f}**")

    # Scatter 3D
    st.markdown("#### Vue 3D : Age, IMC et Charges")
    fig_3d = px.scatter_3d(df, x="age", y="bmi", z="charges", color=color_option,
                            opacity=0.5,
                            labels={"age":"Age","bmi":"IMC","charges":"Frais (USD)"},
                            title="Relation Age, IMC et Frais medicaux",
                            color_discrete_map={"yes":"#ef4444","no":"#22c55e",
                                                "male":"#3b82f6","female":"#f59e0b"})
    fig_3d.update_layout(height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Matrice de correlation
    st.markdown("#### Matrice de correlation (variables numeriques)")
    num_df = df[["age","bmi","children","charges"]].copy()
    corr = num_df.corr().round(3)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu",
                         zmin=-1, zmax=1, title="Matrice de correlation")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 3 : Analyse par groupes ───────────────────────────────────────────────
with tab3:
    st.subheader("Analyse des frais par groupes")

    col_l, col_r = st.columns(2)

    with col_l:
        fig_sex = px.box(df, x="sex", y="charges", color="sex",
                         color_discrete_map={"male":"#3b82f6","female":"#f59e0b"},
                         title="Charges par sexe",
                         labels={"sex":"Sexe","charges":"Frais (USD)"})
        fig_sex.update_layout(showlegend=False)
        st.plotly_chart(fig_sex, use_container_width=True)

    with col_r:
        fig_children = px.box(df, x="children", y="charges",
                               color_discrete_sequence=["#8b5cf6"],
                               title="Charges selon le nombre d'enfants",
                               labels={"children":"Nombre d'enfants","charges":"Frais (USD)"})
        st.plotly_chart(fig_children, use_container_width=True)

    # Heatmap moyenne charges IMC x fumeur
    st.markdown("#### Interaction IMC x Statut fumeur sur les charges moyennes")
    df["bmi_cat"] = pd.cut(df["bmi"], bins=[0,18.5,25,30,35,100],
                           labels=["<18.5","18.5-25","25-30","30-35",">35"])
    pivot = df.groupby(["bmi_cat","smoker"])["charges"].mean().reset_index()
    pivot_wide = pivot.pivot(index="bmi_cat", columns="smoker", values="charges").round(0)
    fig_heat = px.imshow(pivot_wide, text_auto=",.0f",
                         color_continuous_scale="Reds",
                         title="Charges moyennes selon la categorie IMC et le statut fumeur (USD)",
                         labels={"x":"Fumeur","y":"Categorie IMC","color":"Charges (USD)"})
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Les fumeurs avec un IMC eleve constituent le groupe a risque le plus penalise par le modele.")

# ── Tab 4 : Interpretabilite ─────────────────────────────────────────────────
with tab4:
    st.subheader("Interpretabilite des modeles")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Coefficients — Regression Lineaire")
        coef = meta["coef_lr"]
        coef_df = pd.DataFrame({"Variable": list(coef.keys()),
                                 "Coefficient (USD)": list(coef.values())
                                }).sort_values("Coefficient (USD)", ascending=True)
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in coef_df["Coefficient (USD)"]]
        fig_coef = px.bar(coef_df, x="Coefficient (USD)", y="Variable",
                          orientation="h",
                          title=f"Impact de chaque variable sur les charges\n(Intercept : {meta['intercept_lr']:,.0f} USD)",
                          color="Coefficient (USD)",
                          color_continuous_scale=["#3b82f6","#e5e7eb","#ef4444"],
                          range_color=[-5000, 25000])
        fig_coef.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(
            "Lecture : le statut fumeur (smoker_enc) augmente les charges de plus de 23 000 USD en moyenne, "
            "ce qui est le facteur le plus determinant du modele."
        )

    with col_r:
        st.markdown("#### Importance des features — Arbre de Decision")
        imp = meta["feature_importance_dt"]
        imp_df = pd.DataFrame({"Feature": list(imp.keys()),
                                "Importance": list(imp.values())
                               }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Importance relative des features (Arbre de Decision)",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "L'arbre de decision confirme que le statut fumeur est la variable la plus discriminante, "
            "suivi de l'IMC et de l'age."
        )

    # Tableau comparatif des modeles
    st.markdown("#### Comparaison des performances")
    perf_df = pd.DataFrame(meta["metrics"])
    perf_df.columns = ["Modele","MAE (USD)","RMSE (USD)","R²"]
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption(
        "L'Arbre de Decision obtient le meilleur R² (0.865) mais la Regression Lineaire "
        "est privilegiee pour sa transparence et sa conformite RGPD Art. 22."
    )
