import streamlit as st
from openai import OpenAI
import re
import json
from vad_to_music import vad_to_music

# Configuration de l'API OpenAI (remplace "your_api_key_here" par ta clé)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# Titre de l'application
st.title("Musicothérapie for SHIA 🎵🧠")

# Introduction
st.write("Bienvenue dans cette étude de musicothérapie. Discutez avec une IA spécialisée pour analyser votre état émotionnel à travers la musique.")

# Initialiser l'historique de la conversation dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stocker vad_data.json
if "vad_data" not in st.session_state:
    st.session_state.vad_data = {}


# Affichage des messages précédents
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if len (st.session_state.messages) == 0:

    # Preprompt
    pre_prompt = [
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": "You are a qualified psychologist specializing in music therapy.  I want you to ask me three questions (not scored) regarding my mood to determine my VAD values (Valence, Arousal and dominance). Don't ask them in one prompt but three to make it so that you can tune the questions to be more precise. The questions SHOULD NOT ASK FOR A SCORE FROM THE USER. It should be a smooth approach (don't enumerate the questions) and work like a funnel to grasp any emotion. Each score should be between 0 and ten. The distributions of our data can be considered normal for the three dimensions and centered around 6.5 for valence, 4 for arousal and 6 for dominance. Give the three values with one decimal place. After my third answer, please give me the scores in a json format as follow output_prompt = {    'valence': 2.1,   'arousal': 4.4,  'dominance': 6.8}, without any other text (don't say things such as here is the json). Remember, ask no more than three question and then only return the JSON. Start right away with the first question after saying hi."
            }
        ]
        }
    ]


    response = client.responses.create(
        model="gpt-4o-mini",
        input=pre_prompt,
        text={
            "format": {
            "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=615,
        top_p=1,
        store=True
    )

    assistant_message = response.output[0].content[0].text

    st.session_state.messages.append({"role": "system", "content": pre_prompt[0]["content"][0]["text"]}) 
    st.session_state.messages.append({"role": "assistant", "content": assistant_message}) 

    with st.chat_message("assistant"):
        st.markdown(assistant_message)

# Entrée utilisateur
user_input = st.text_input("Votre réponse :", key="user_input")


# Traitement de la réponse utilisateur
if st.button("Envoyer") and user_input:

    # Ajouter le message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Afficher le message utilisateur dans l'interface
    with st.chat_message("user"):
        st.markdown(user_input)

    # Appel à l'API OpenAI
    response = client.responses.create(
        model="gpt-4o-mini",
        input=st.session_state.messages,
        text={
    "format": {
    "type": "text"
    }
},
        reasoning={},
        tools=[],
        temperature=1, # Niveau de créativité, 0 = très conservateur, 1 = très créatif
        max_output_tokens=615,
        top_p=1, # Seuil de probabilité pour le sampling (nucleus sampling), 0 = pas de sampling (réponses conservatives), 1 = sampling complet (sert à éviter les réponses trop répétitives)
        store=True # Réponse sauvegardée pour analyse
)


    # Récupérer le message généré
    assistant_message = response.output[0].content[0].text

    if "{" in assistant_message:

        st.session_state.vad_data = assistant_message
        st.markdown("Thank you for your participation.")

    else:
        # Ajouter le message de l'IA à l'historique
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})

        # Afficher le message de l'IA dans l'interface
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
            

if st.session_state.vad_data:
    data_str = st.session_state.vad_data  # la chaîne d'entrée

    # Chercher le bloc JSON délimité par ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", data_str, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Sinon, on suppose que data_str est déjà du JSON pur.
        json_str = data_str.strip()  # enlever d'éventuels espaces inutiles

    try:
        vad_dict = json.loads(json_str)
        link = vad_to_music(vad_dict)
        st.markdown("Music successfully found!")
        st.write(link)

    except json.JSONDecodeError as e:
        st.error("Error with JSON conversion : " + str(e))



# bash command to run streamlit app
# streamlit run streamlit_app.py
# bash command to stop streamlit app
# ctrl + c