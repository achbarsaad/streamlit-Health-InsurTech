from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

with open(Path(__file__).parent.parent / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not st.session_state.get("authenticated", False):
    st.warning("Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

from utils.loader import load_model, load_meta, load_silver
logger.info(f"Page Prediction — user : {st.session_state.get('username','?')}")

meta  = load_meta()
df    = load_silver()
enc   = meta["encoders"]

# Charger les 3 modeles en backend
lr    = load_model("linear_regression")
dt    = load_model("decision_tree")
ridge = load_model("ridge")
MODELS = {
    "Regression Lineaire  (R²=0.78, transparent, conforme RGPD Art.22)": lr,
    "Arbre de Decision    (R²=0.87, regles lisibles)": dt,
    "Ridge                (R²=0.78, robuste aux biais)": ridge,
}

st.title("Simulation des frais medicaux")
st.markdown(
    "Renseignez vos caracteristiques ci-dessous. "
    "La prediction est une **estimation indicative** et ne constitue pas une offre contractuelle."
)
st.divider()

left, right = st.columns([1, 2], gap="large")

# ── Formulaire ────────────────────────────────────────────────────────────────
with left:
    st.subheader("Vos informations")

    age      = st.slider("Age (ans)", 18, 64, 35,
                          help="Votre age en annees")
    sex      = st.selectbox("Sexe", ["male","female"],
                             format_func=lambda x: "Homme" if x=="male" else "Femme")
    bmi      = st.slider("IMC — Indice de Masse Corporelle",
                          15.0, 55.0, 28.0, 0.1,
                          help="Poids (kg) / Taille² (m). Normal : 18.5 a 24.9")
    children = st.slider("Nombre de personnes a charge", 0, 5, 0)
    smoker   = st.selectbox("Statut fumeur", ["no","yes"],
                              format_func=lambda x: "Non-fumeur" if x=="no" else "Fumeur")
    region   = st.selectbox("Region", sorted(enc["region"].keys()),
                              help="Region de residence")

    model_label = st.selectbox("Modele de prediction", list(MODELS.keys()))

    # Validation des entrees
    errors = []
    if not (18 <= age <= 64):
        errors.append("L'age doit etre entre 18 et 64 ans.")
    if not (10.0 <= bmi <= 60.0):
        errors.append("L'IMC doit etre entre 10 et 60.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    predict_btn = st.button("Simuler mes frais medicaux", type="primary",
                             use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
with right:
    st.subheader("Resultat de la simulation")

    if predict_btn:
        # Construction du vecteur de features
        input_vec = pd.DataFrame([{
            "age":        age,
            "sex_enc":    enc["sex"][sex],
            "bmi":        bmi,
            "children":   children,
            "smoker_enc": enc["smoker"][smoker],
            "region_enc": enc["region"][region],
        }])

        selected_model = MODELS[model_label]
        prediction     = float(selected_model.predict(input_vec)[0])
        prediction      = max(prediction, 0)  # pas de frais negatifs

        logger.info(
            f"Prediction — user:{st.session_state.get('username','?')} "
            f"age:{age} bmi:{bmi:.1f} smoker:{smoker} -> {prediction:,.0f} USD"
        )

        # Intervalle de confiance approximatif (+/- 1 MAE)
        model_key = model_label.split("(")[0].strip().lower().replace(" ","_")
        mae_model = next((m["MAE"] for m in meta["metrics"]
                          if m["model"].lower().replace(" ","_") in model_key.replace(" ","_")),
                         meta["metrics"][0]["MAE"])
        ci_low  = max(0, prediction - mae_model)
        ci_high = prediction + mae_model

        # Affichage du resultat principal
        st.markdown(
            f"""
            <div style="background:#dbeafe;border-left:5px solid #2563eb;
                        padding:20px;border-radius:8px;margin-bottom:16px;">
                <h2 style="color:#1e40af;margin:0;">Estimation : {prediction:,.0f} USD / an</h2>
                <p style="color:#1e3a8a;margin:6px 0 0 0;">
                    Intervalle indicatif : {ci_low:,.0f} — {ci_high:,.0f} USD
                    (± {mae_model:,.0f} USD, MAE du modele)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Metriques comparatives
        c1, c2, c3 = st.columns(3)
        pct = (df["charges"] < prediction).mean() * 100
        c1.metric("Votre estimation", f"{prediction:,.0f} USD")
        c2.metric("Mediane du dataset", f"{df['charges'].median():,.0f} USD")
        c3.metric("Votre percentile", f"{pct:.0f}e")

        st.divider()

        # Graphique : position dans la distribution
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_dist = px.histogram(df, x="charges", nbins=50, opacity=0.6,
                                    title="Votre estimation dans la distribution globale",
                                    labels={"charges":"Frais (USD)"},
                                    color_discrete_sequence=["#93c5fd"])
            fig_dist.add_vline(x=prediction, line_dash="solid", line_color="#2563eb",
                                line_width=3,
                                annotation_text=f"Vous : {prediction:,.0f}",
                                annotation_position="top right")
            fig_dist.add_vrect(x0=ci_low, x1=ci_high,
                                fillcolor="#2563eb", opacity=0.1,
                                annotation_text="Intervalle", annotation_position="top left")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_g2:
            # Contribution de chaque facteur (coefficients LR)
            coef     = meta["coef_lr"]
            intercept = meta["intercept_lr"]
            contribs = {
                "Constante":    intercept,
                "Age":          coef["age"] * age,
                "Sexe":         coef["sex_enc"] * enc["sex"][sex],
                "IMC":          coef["bmi"] * bmi,
                "Enfants":      coef["children"] * children,
                "Fumeur":       coef["smoker_enc"] * enc["smoker"][smoker],
                "Region":       coef["region_enc"] * enc["region"][region],
            }
            contrib_df = pd.DataFrame({
                "Facteur": list(contribs.keys()),
                "Contribution (USD)": [round(v, 0) for v in contribs.values()]
            }).sort_values("Contribution (USD)", ascending=True)

            fig_contrib = px.bar(
                contrib_df, x="Contribution (USD)", y="Facteur", orientation="h",
                title="Decomposition de l'estimation (modele lineaire)",
                color="Contribution (USD)",
                color_continuous_scale=["#3b82f6","#e5e7eb","#ef4444"],
                range_color=[-3000, 25000]
            )
            fig_contrib.update_layout(coloraxis_showscale=False)
            fig_contrib.add_vline(x=0, line_color="black", line_width=0.8)
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.caption(
                "La decomposition est basee sur la Regression Lineaire, "
                "independamment du modele choisi pour la prediction principale."
            )

        # Tableau recapitulatif
        st.markdown("#### Recapitulatif de votre profil")
        recap = pd.DataFrame({
            "Parametre": ["Age","Sexe","IMC","Enfants","Fumeur","Region","Modele utilise"],
            "Valeur": [f"{age} ans",
                       "Homme" if sex=="male" else "Femme",
                       f"{bmi:.1f}",
                       str(children),
                       "Oui" if smoker=="yes" else "Non",
                       region,
                       model_label.split("(")[0].strip()]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)

        # Mention ethique
        st.info(
            "Cette estimation a ete calculee a partir de donnees anonymisees et d'un modele "
            "interpretable. Aucune donnee personnelle n'est stockee. "
            "Ce resultat est fourni a titre indicatif uniquement — consultez un conseiller "
            "pour un devis contractuel."
        )

    else:
        st.info("Renseignez votre profil dans le formulaire a gauche et cliquez sur 'Simuler'.")

        # Contexte
        st.markdown("#### Contexte : qui paie le plus ?")
        col1, col2 = st.columns(2)
        with col1:
            grp = df.groupby("smoker")["charges"].mean().reset_index()
            grp["smoker"] = grp["smoker"].map({"yes":"Fumeurs","no":"Non-fumeurs"})
            fig = px.bar(grp, x="smoker", y="charges",
                         color="smoker",
                         color_discrete_map={"Fumeurs":"#ef4444","Non-fumeurs":"#22c55e"},
                         title="Charges moyennes : fumeurs vs non-fumeurs",
                         labels={"smoker":"","charges":"USD / an"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp2 = df.groupby("region")["charges"].mean().sort_values(ascending=False).reset_index()
            fig2 = px.bar(grp2, x="region", y="charges",
                          color="charges", color_continuous_scale="Blues",
                          title="Charges moyennes par region",
                          labels={"region":"Region","charges":"USD / an"})
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
