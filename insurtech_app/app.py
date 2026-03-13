from pathlib import Path
import streamlit as st
import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Health-InsurTech",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open(Path(__file__).parent / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── RGPD consent banner ───────────────────────────────────────────────────────
if "rgpd_accepted" not in st.session_state:
    st.session_state["rgpd_accepted"] = False

if not st.session_state["rgpd_accepted"]:
    st.markdown("""
    <div style="background:#fef3c7;border:2px solid #f59e0b;border-radius:8px;padding:20px;margin-bottom:20px;">
    <h4 style="color:#92400e;margin:0 0 10px 0;">Notice de confidentialite et consentement RGPD</h4>
    <p style="color:#78350f;margin:0 0 10px 0;">
    Cette application collecte et traite des donnees a caractere personnel dans le but exclusif
    de simuler des frais medicaux annuels. Les donnees saisies ne sont pas stockees de maniere
    permanente. Conformement au RGPD (UE 2016/679), vous disposez d'un droit d'acces,
    de rectification et de suppression de vos donnees. Aucune donnee directement identifiante
    n'est utilisee dans les modeles de prediction.
    </p>
    <p style="color:#78350f;font-size:0.85em;margin:0;">
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("J'accepte", type="primary", use_container_width=True):
            st.session_state["rgpd_accepted"] = True
            logger.info("Consentement RGPD accepte")
            st.rerun()
    with col2:
        if st.button("Refuser", use_container_width=False):
            st.error("Vous devez accepter la politique de confidentialite pour utiliser cette application.")
    st.stop()

# ── Authentification ─────────────────────────────────────────────────────────
def check_credentials(user, pwd):
    try:
        return (user.strip() == st.secrets["auth"]["admin_user"] and
                pwd == st.secrets["auth"]["admin_password"])
    except Exception:
        return user.strip() == "admin" and pwd == "insurtech2024"

if not st.session_state.get("authenticated", False):
    st.title("Connexion")
    with st.form("login"):
        user = st.text_input("Nom d'utilisateur")
        pwd  = st.text_input("Mot de passe", type="password")
        ok   = st.form_submit_button("Se connecter", type="primary")
    if ok:
        if not user or not pwd:
            st.error("Veuillez remplir tous les champs.")
        elif len(user) > 50 or len(pwd) > 100:
            st.error("Entrees invalides.")
        elif check_credentials(user, pwd):
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            logger.info(f"Connexion reussie : {user}")
            st.rerun()
        else:
            st.error("Identifiants incorrects.")
            logger.warning(f"Echec connexion : {user}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"Connecte : **{st.session_state.get('username','')}**")
    if st.button("Se deconnecter"):
        for k in ["authenticated","username","rgpd_accepted"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Page accueil ──────────────────────────────────────────────────────────────
logger.info(f"Accueil — user : {st.session_state.get('username','?')}")

st.title("Health-InsurTech — Estimateur de frais medicaux")
st.markdown("**Outil de simulation tarifaire ethique et transparent**")
st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
### A propos de cette application

Health-InsurTech est un outil de simulation des frais medicaux annuels developpe
pour accompagner les futurs assures dans le choix de leur contrat de sante.
Il est concu selon les principes **Ethic-by-Design** et est pleinement conforme au **RGPD**.

#### Donnees utilisees
Le modele a ete entraine sur un jeu de donnees de **1 338 clients** anonymises,
contenant les variables suivantes apres suppression des donnees directement identifiantes :

- **Age** — age de l'assure (18 a 64 ans)
- **Sexe** — homme ou femme
- **IMC (BMI)** — indice de masse corporelle
- **Enfants** — nombre de personnes a charge
- **Statut fumeur** — oui ou non
- **Region** — zone geographique de residence

#### Modeles disponibles
Trois modeles interpretables ont ete entraines et sont disponibles a la prediction :
- **Regression Lineaire** — coefficients directement lisibles, conforme Art. 22 RGPD
- **Arbre de Decision** — regles explicites, profondeur limitee a 4
- **Ridge** — regression regularisee, robuste aux biais

#### Garanties ethiques
Les donnees directement identifiantes (nom, prenom, email, telephone, numero de securite sociale,
adresse IP) ont ete supprimees avant tout traitement. La prediction fournie est une
**estimation indicative** et ne constitue pas une offre contractuelle.
    """)

with col2:
    from utils.loader import load_meta
    meta = load_meta()
    stats = meta["df_stats"]

    st.markdown("#### Statistiques du jeu de donnees")
    st.metric("Clients dans le dataset", f"{stats['n_rows']:,}")
    st.metric("Frais moyens (ensemble)", f"{stats['charges_mean']:,.0f} USD")
    st.metric("Frais medians", f"{stats['charges_median']:,.0f} USD")
    st.metric("Frais moyens — Fumeurs", f"{stats['smoker_mean']:,.0f} USD")
    st.metric("Frais moyens — Non-fumeurs", f"{stats['non_smoker_mean']:,.0f} USD")
    st.metric("Ratio fumeur / non-fumeur",
              f"{stats['smoker_mean']/stats['non_smoker_mean']:.1f}x")

    st.markdown("#### Performances des modeles (jeu de test)")
    import pandas as pd
    perf_df = pd.DataFrame(meta["metrics"])
    perf_df.columns = ["Modele","MAE (USD)","RMSE (USD)","R²"]
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
**Navigation** — Utilisez le menu de gauche pour acceder aux pages :
- **Donnees** — Explorer et telecharger les donnees bronze et silver
- **Visualisations** — Graphiques interactifs sur les correlations et distributions
- **Prediction** — Simuler vos frais medicaux en temps reel
""")
