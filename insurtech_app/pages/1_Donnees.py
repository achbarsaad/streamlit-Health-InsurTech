from pathlib import Path
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Donnees", page_icon="📁", layout="wide")

with open(Path(__file__).parent.parent / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not st.session_state.get("authenticated", False):
    st.warning("Veuillez vous connecter depuis la page d'accueil.")
    st.stop()

from utils.loader import load_bronze, load_silver, load_encoded, to_csv_bytes
logger.info(f"Page Donnees — user : {st.session_state.get('username','?')}")

st.title("Donnees — Exploration et telechargement")
st.markdown("""
Les donnees sont organisees selon une architecture **Bronze / Silver** :
- **Bronze** : donnees brutes telles que recues, incluant toutes les colonnes originales
- **Silver** : donnees anonymisees (suppression des colonnes RGPD) et encodees pour la modelisation
""")
st.divider()

tab_bronze, tab_silver, tab_encoded = st.tabs([
    "Bronze — Donnees brutes",
    "Silver — Anonymisees",
    "Silver — Encodees (modele)"
])

# ── Bronze ────────────────────────────────────────────────────────────────────
with tab_bronze:
    df_b = load_bronze()
    st.markdown(f"**{df_b.shape[0]:,} lignes x {df_b.shape[1]} colonnes** — Donnees brutes completes")

    st.warning(
        "Ce fichier contient des donnees directement identifiantes (nom, prenom, email, "
        "telephone, numero de securite sociale, adresse IP). "
        "Il ne doit jamais etre utilise directement dans un modele de Machine Learning "
        "conformement au principe de minimisation des donnees (RGPD Art. 5)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lignes", f"{df_b.shape[0]:,}")
    col2.metric("Colonnes", str(df_b.shape[1]))
    col3.metric("Colonnes PII", "15")
    col4.metric("Valeurs manquantes", str(df_b.isnull().sum().sum()))

    n = st.slider("Nombre de lignes a afficher", 5, 50, 10, key="bronze_rows")
    st.dataframe(df_b.head(n), use_container_width=True)

    st.markdown("**Types et valeurs manquantes**")
    info = pd.DataFrame({
        "Type": df_b.dtypes.astype(str),
        "Non-nuls": df_b.count(),
        "Nuls": df_b.isnull().sum(),
        "Uniques": df_b.nunique()
    })
    st.dataframe(info, use_container_width=True)

    st.download_button(
        "Telecharger bronze (CSV)",
        data=to_csv_bytes(df_b),
        file_name="insurance_bronze.csv",
        mime="text/csv"
    )

# ── Silver anonymisé ─────────────────────────────────────────────────────────
with tab_silver:
    df_s = load_silver()
    st.markdown(f"**{df_s.shape[0]:,} lignes x {df_s.shape[1]} colonnes** — Apres suppression des 15 colonnes RGPD")
    st.success(
        "Toutes les colonnes directement identifiantes ont ete supprimees. "
        "Ce fichier contient uniquement les variables analytiques necessaires a la prediction."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", f"{df_s.shape[0]:,}")
    col2.metric("Colonnes conservees", str(df_s.shape[1]))
    col3.metric("Colonnes supprimees", "15")

    st.markdown("**Colonnes conservees**")
    col_info = {
        "age": "Age de l'assure (18-64 ans)",
        "sex": "Sexe (male / female)",
        "bmi": "Indice de masse corporelle",
        "children": "Nombre d'enfants a charge",
        "smoker": "Statut fumeur (yes / no)",
        "region": "Region de residence (US)",
        "charges": "Frais medicaux annuels — CIBLE",
        "sex_enc": "Sexe encode numeriquement",
        "smoker_enc": "Fumeur encode numeriquement",
        "region_enc": "Region encodee numeriquement",
    }
    st.dataframe(
        pd.DataFrame({"Colonne": list(col_info.keys()), "Description": list(col_info.values())}),
        use_container_width=True, hide_index=True
    )

    n2 = st.slider("Nombre de lignes", 5, 50, 10, key="silver_rows")
    st.dataframe(df_s.head(n2), use_container_width=True)

    st.markdown("**Statistiques descriptives**")
    st.dataframe(df_s.describe().round(2).T, use_container_width=True)

    st.download_button(
        "Telecharger silver anonymisee (CSV)",
        data=to_csv_bytes(df_s),
        file_name="insurance_silver_anonymized.csv",
        mime="text/csv"
    )

# ── Silver encodé ─────────────────────────────────────────────────────────────
with tab_encoded:
    df_e = load_encoded()
    st.markdown(f"**{df_e.shape[0]:,} lignes x {df_e.shape[1]} colonnes** — Donnees pret-a-modeliser")
    st.info(
        "Ce fichier est celui utilise directement pour l'entrainement des modeles. "
        "Les variables categorielles ont ete encodees en valeurs numeriques via LabelEncoder."
    )

    from utils.loader import load_meta
    meta = load_meta()
    enc = meta["encoders"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Encodage 'sex'**")
        st.dataframe(pd.DataFrame(enc["sex"].items(), columns=["Valeur","Code"]),
                     hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Encodage 'smoker'**")
        st.dataframe(pd.DataFrame(enc["smoker"].items(), columns=["Valeur","Code"]),
                     hide_index=True, use_container_width=True)
    with col3:
        st.markdown("**Encodage 'region'**")
        st.dataframe(pd.DataFrame(enc["region"].items(), columns=["Valeur","Code"]),
                     hide_index=True, use_container_width=True)

    n3 = st.slider("Nombre de lignes", 5, 50, 10, key="encoded_rows")
    st.dataframe(df_e.head(n3), use_container_width=True)

    st.download_button(
        "Telecharger silver encodee (CSV)",
        data=to_csv_bytes(df_e),
        file_name="insurance_silver_encoded.csv",
        mime="text/csv"
    )
