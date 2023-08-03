"""usage: streamlit run audiobot.py"""
import os
import tempfile

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore, Weaviate

# URL API endpoints obtained from the Prem App UI
openai.api_key = os.environ["OPENAI_API_KEY"] = "random-string"
whisper_url = "http://127.0.0.1:10111/v1"  # audio-to-text
embeddings = OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8444/v1")  # All MiniLM L6 v2
weaviate_url = "http://127.0.0.1:8080"  # vector store
vicuna_api = "http://127.0.0.1:8111/v1"  # LLM


def convert_audio_to_text(audio_file_path) -> str:
    openai.api_base = whisper_url
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.get("text")


def create_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
    return text_splitter.create_documents([text])


def add_to_vectorstore(texts) -> VectorStore:
    documents = [Document(page_content=t.page_content) for t in texts]
    return Weaviate.from_documents(documents, embeddings, weaviate_url=weaviate_url, by_text=False)


def query_vectorstore(query: str, vectorstore: VectorStore) -> str:
    query_vector = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_vector, k=1)
    return ", ".join(doc.page_content for doc in docs)


chat = ChatOpenAI(openai_api_base=vicuna_api, max_tokens=256)
st.title("Chat with audio files")
with st.sidebar:
    upload_file = st.file_uploader("Choose an audio file")
    if upload_file is not None:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(upload_file.read())
            st.write("Converting audio to text...")
            text = convert_audio_to_text(tmp.name)
        st.write("... done!")
        chunks = create_chunks(text)
        vector_db = add_to_vectorstore(chunks)

user_input = st.text_input("Enter your question here...")
if user_input:
    context = query_vectorstore(user_input, vector_db)
    content = f"Use the context below to answer the Question.\n\nContext: {context}.\n\nQuestion: {user_input}"
    messages = [HumanMessage(content=content)]
    st.write("Generating...")
    res = chat(messages)
    st.write(res.content)
