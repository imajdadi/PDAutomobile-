import streamlit as st
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import os
from huggingface_hub import login

# Charger les variables d'environnement
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("⚠️ La clé API Hugging Face est manquante.")
    st.stop()

# Connexion Hugging Face
login(hf_token)

# Configuration de la page
st.set_page_config(page_title="Diagnostic Auto", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv('https://huggingface.co/spaces/majdaImane/PDAutomobile/resolve/main/commentaires_avec_pannes_combinees.csv')

df = load_data()

st.title("🔧 Plateforme Diagnostic Automobile")

col1, col2 = st.columns([1, 2])

# Choix du profil
with col1:
    st.header("Choisissez votre profil")
    profil = st.radio("Vous êtes :", ["🚗 Client", "🔧 Pro de la casse"])

# Choix véhicule
with col2:
    st.header("Informations véhicule")
    marques = df["marque"].dropna().unique()
    marque_selection = st.selectbox("Sélectionnez la marque", marques)

    modeles = df[df["marque"] == marque_selection]["modele"].dropna().unique()
    modele_selection = st.selectbox("Sélectionnez le modèle", modeles)

    pannes = df[(df["marque"] == marque_selection) & (df["modele"] == modele_selection)]["pannes_combinees"].values

    if profil == "🚗 Client":
        st.subheader("🔍 Pannes fréquentes à connaître")
        if pannes:
            liste_pannes = [p.strip() for p in pannes[0].strip("[]").replace("'", "").split(",")]
            pannes_str = " | ".join(liste_pannes)
            st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;font-size:18px;'>"
                        f"{pannes_str}</div>", unsafe_allow_html=True)
        else:
            st.warning("Aucune information sur les pannes pour ce modèle.")

    elif profil == "🔧 Pro de la casse":
        st.subheader("♻️ Idées de pièces à recycler")
        if pannes:
            pieces = [p.strip() for p in pannes[0].strip("[]").replace("'", "").split(",")]
            st.markdown("Proposez à la vente les pièces liées aux problèmes suivants :")
            st.markdown("<ul style='list-style-type:none;'>", unsafe_allow_html=True)
            for piece in pieces:
                st.markdown(f"<li style='font-size:16px;'>✅ {piece}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.warning("Aucune panne détectée pour ce modèle.")# 🔮 SECTION CHATBOT INTELLIGENT

st.divider()
st.header("💬 Posez une question à notre assistant intelligent !")

@st.cache_resource
def load_chatbot():
    return pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        tokenizer="HuggingFaceH4/zephyr-7b-beta",
        device=0,
        torch_dtype="auto"
    )

chatbot = load_chatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Votre question :")

if user_input:
    if profil == "🚗 Client":
        system_prompt = (
            "Tu es un expert automobile spécialisé dans l'achat de voitures d'occasion. "
            "Aide le client à identifier les risques liés aux pannes fréquentes et donne des conseils d'inspection."
        )
    else:  # profil == "🔧 Pro de la casse"
        system_prompt = (
            "Tu es un expert en recyclage automobile. "
            "Aide le professionnel de la casse à identifier les pièces les plus demandées et récupérables en fonction des pannes fréquentes."
        )

    # Récupération et formatage des pannes
    if pannes:
        pannes_liste = [p.strip() for p in pannes[0].strip("[]").replace("'", "").split(",")]
        toutes_les_pannes = "; ".join(pannes_liste)
    else:
        toutes_les_pannes = "aucune panne connue"

    prompt = (
        f"<|system|>{system_prompt} Le véhicule concerné est une {marque_selection} {modele_selection}. "
        f"Les pannes fréquentes sont : {toutes_les_pannes}.\n"
        f"<|user|>{user_input}\n"
        f"<|assistant|>"
    )

    with st.spinner("Réflexion de l'assistant... 🤖"):
        response = chatbot(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
        reponse_utilisateur = response.split("<|assistant|>")[-1].strip()

    st.session_state.chat_history.append((user_input, reponse_utilisateur))

# Affichage de l'historique
st.subheader("📜 Historique du Chat")
for question, answer in reversed(st.session_state.chat_history):
    st.markdown(f"**Vous :** {question}")
    st.markdown(f"**Assistant :** {answer}")
    st.markdown("---")

# Réinitialisation du chat
if st.button("🔄 Réinitialiser le chat"):
    st.session_state.chat_history = []
    st.success("Chat réinitialisé avec succès.")
