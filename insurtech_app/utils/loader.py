from pathlib import Path
import pandas as pd
import pickle
import streamlit as st

ROOT = Path(__file__).parent.parent

@st.cache_data
def load_bronze():
    return pd.read_csv(ROOT / "data" / "bronze" / "insurance_data.csv")

@st.cache_data
def load_silver():
    return pd.read_csv(ROOT / "data" / "silver" / "insurance_anonymized.csv")

@st.cache_data
def load_encoded():
    return pd.read_csv(ROOT / "data" / "silver" / "insurance_encoded.csv")

@st.cache_resource
def load_model(name: str):
    with open(ROOT / "models" / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_meta():
    with open(ROOT / "models" / "meta.pkl", "rb") as f:
        return pickle.load(f)

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
