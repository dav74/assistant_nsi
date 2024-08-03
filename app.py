from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import streamlit as st

api_key_llm = st.secrets["GROQ_API_KEY"]
api_key_embedding = st.secrets["HF_API_KEY"]
llm = Groq(temperature=0.8,model="llama3-70b-8192", api_key=api_key_llm)
embed_model = HuggingFaceInferenceAPIEmbedding(model_name="OrdalieTech/Solon-embeddings-large-0.1", token = api_key_embedding)
Settings.llm = llm
Settings.embed_model = embed_model
storage_context = StorageContext.from_defaults(persist_dir='./index_storage')
vector_index = load_index_from_storage(storage_context)
st.set_page_config(page_title="ASSISTANT NSI (première et terminale)", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Assistant NSI")
st.info("Cet assistant vous permet de réviser le cours et faire des exercices.")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour, que puis-je faire pour vous ?"}
    ]
if "chat_engine" not in st.session_state.keys(): 
    st.session_state.chat_engine = vector_index.as_chat_engine(
    chat_mode = "context", streaming=True,
    system_prompt = (
        "Tu es un assistant spécialisé dans l'enseignement de la spécialité Numérique et sciences informatiques en classe de première et de terminal"
        'Tu as un bon niveau en langage Python'
        'Tu dois commencer la conversation'
        "Inspire-toi des sujets de bac donnés en exemple pour créer des exercices"
        "Inspire-toi des sujets d'épreuve pratique pour créer des exercices sur la programmation en Python"
        'Ton interlocuteur est un élève qui suit la spécialité nsi en première et en terminale'
        'Tu dois uniquement répondre aux questions qui concernent la spécialité numérique et sciences informatiques'
        "Tu ne dois pas faire d'erreur, répond à la question uniquement si tu es sûr de ta réponse"
        "si tu ne trouves pas la réponse à une question, tu réponds que tu ne connais pas la réponse et que l'élève doit s'adresser à son professeur pour obtenir cette réponse"
        "Tu dois uniquement aborder des notions qui sont aux programmes de la spécialité numérique et sciences informatiques (première et terminale), tu ne dois jamais aborder une notion qui n'est pas au programme"
        'Tu dois uniquement répondre en langue française'
        'les réponses doivent être données au format Markdown'
    )
)
prompt = st.chat_input("À vous...")
if prompt :
    st.session_state.messages.append({"role": "user", "content": prompt})
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)