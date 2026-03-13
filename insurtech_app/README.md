# Health-InsurTech

Application Streamlit de simulation des frais medicaux.

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

Identifiants : admin / insurtech2024

## Structure
- data/bronze/  : donnees brutes originales
- data/silver/  : donnees anonymisees et encodees (RGPD)
- models/       : modeles pre-entraines (Linear, Decision Tree, Ridge)
- pages/        : 1_Donnees, 2_Visualisations, 3_Prediction
