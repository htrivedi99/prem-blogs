import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

import tempfile
import streamlit as st

os.environ["OPENAI_API_KEY"] = "random-string"

embeddings = OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8444/v1")
weaviate_url = "http://127.0.0.1:8080"

def convert_audio_to_text(audio_file_path) -> str:
    openai.api_base = "http://127.0.0.1:10111/v1"
    openai.api_key = "random-string"

    audio_file = open(audio_file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.get('text')


def convert_text_to_embedding(text: str):
    doc_result = embeddings.embed_documents([text])
    embedding = doc_result[0]
    return embedding

def create_chunks(text_blob: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )

    texts = text_splitter.create_documents([text_blob])
    return texts


def add_to_vectorstore(texts):
    embeddings = OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8444/v1")
    documents = []
    for t in texts:
        doc = Document(page_content=t.page_content)
        documents.append(doc)

    vectorstore = Weaviate.from_documents(
        documents,
        embeddings,
        weaviate_url=weaviate_url,
        by_text=False,
    )
    return vectorstore


def query_vectorstore(query: str, vectorstore):
    embeddings = OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8444/v1")
    context = ""
    query_vector = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_vector, k=1)
    for doc in docs:
        context += doc.page_content
    return context


st.title('Chat with audio files')
with st.sidebar:
    upload_file = st.file_uploader("Choose a file")
    if upload_file is not None:
        print(upload_file)
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(upload_file.read())
            text_blob = convert_audio_to_text(tmp.name)

            print("text blob converted!")
            chunks = create_chunks(text_blob)
            print("created chunks!")
            vector_db = add_to_vectorstore(chunks)
            print("added to vector db!")

user_input = st.text_input('Enter your question here...')
if user_input:

    context = query_vectorstore(user_input, vector_db)
    print("context: \n")
    print(context)
    print("\n")

    chat = ChatOpenAI(openai_api_base="http://127.0.0.1:8111/v1", max_tokens=256)

    content = "Use the context below to answer the question.\n" \
              f"Context: {context}. \n" \
              f"Question: {user_input}"

    messages = [
        HumanMessage(content=content)
    ]
    print("generating...")
    res = chat(messages)
    print(res)
    st.write(res.content)







