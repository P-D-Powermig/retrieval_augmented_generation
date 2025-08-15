import streamlit as st
import os
from dotenv import load_dotenv
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from htmlTemplates import user_template, bot_template, css

TF_ENABLE_ONEDNN_OPTS=0

# Dicionário de modelos de embedding disponíveis
EMBEDDING_MODELS = {
    "BAAI/bge-m3": "BAAI/bge-m3",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "nomic-embedding-text": "nomic-ai/nomic-embed-text-v1",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2": "intfloat/e5-large-v2",
    "gte-multilingual-base": "thenlper/gte-base",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
}
#### extract_text
def extract_text_and_tables_from_uploaded_pdfs(uploaded_files, temp_dir="temp_pdfs"):
    os.makedirs(temp_dir, exist_ok=True)
    all_text = ""
    all_tables = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extrair texto com LangChain (usa pdfminer)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            all_text += doc.page_content + "\n"

        # Extrair tabelas com pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # page.to_image().draw_lines().draw_text().save("debug_page.png")
                tables = page.extract_tables()
                for table in tables:
                    all_tables.append(table)

    print("Texto extraído:\n", all_text)
    print("\nTabelas extraídas:")
    for i, table in enumerate(all_tables):
        print(f"\nTabela {i+1}:")
        for row in table:
            print(row)

    return all_text, all_tables


def extract_text_from_pdfs(pdf_paths):
    all_text = ""
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()  # Cada item representa uma página
        for doc in documents:
            all_text += doc.page_content + "\n"
    print(all_text)
    return all_text

def extract_text_from_uploaded_pdfs(uploaded_files, temp_dir="temp_pdfs"):
    os.makedirs(temp_dir, exist_ok=True)
    all_text = ""

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            all_text += doc.page_content + "\n"

    print(f"Todo o texto \/ \n{all_text}")
    return all_text

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text

#### chunks
def split_text_into_chunks(text):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "\.", ""]
    )

    c_splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    print(f"Chunks \/ \n{r_splitter.split_text(text)}")

    return r_splitter.split_text(text)

#### vectorstore
def create_vectorstore(text_chunks, model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)

    return vectorstore

#### chain
def build_conversational_chain(vectorstore):
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key="answer")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True
    )
    return chain

def display_chat_messages():
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='RAG test', page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat com o PowerRobot :robot_face:")

    # Sidebar para upload e seleção de embedding
    with st.sidebar:
        st.subheader("Seus documentos")

        selected_embedding_model = st.selectbox(
            "Escolha o modelo de embedding",
            options=list(EMBEDDING_MODELS.keys()),
            index=0
        )

        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        if st.button("Confirmar") and pdf_docs:
            with st.spinner("Processando documentos..."):
                text = extract_text_from_pdfs(pdf_docs)
                # text, _ = extract_text_and_tables_from_uploaded_pdfs(pdf_docs)
                # text = extract_text_from_uploaded_pdfs(pdf_docs)
                text_chunks = split_text_into_chunks(text)
                model_path = EMBEDDING_MODELS[selected_embedding_model]
                vectorstore = create_vectorstore(text_chunks, model_path)
                st.session_state.conversation = build_conversational_chain(vectorstore)
                st.success("Documentos processados com sucesso!")

    # Exibe histórico de conversa
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Input fixo no rodapé
    if st.session_state.conversation:
        user_input = st.chat_input("Digite sua pergunta sobre o documento:")
        if user_input:
            response = st.session_state.conversation({"question": user_input})
            st.session_state.chat_history = response['chat_history']
            st.rerun()

if __name__ == '__main__':
    main()
