import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from htmlTemplates import user_template, bot_template, css

## frameworks
# langchain
# ollama
# crewai
# embedchain

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    ##### embeddings model
    # BAAI/bge-m3 (multi)
    # BAAI/bge-base-en-v1.5
    # BAAI/bge-large-en-v1.5
    # nomic-embedding-text 
    # all-MiniLM-L6-v2 
    # intfloat/e5-large-v2 (port)
    # gte-multilingual-base
    # all-mpnet-base-v2


    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def build_conversational_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini")

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


import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title='RAG test', page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    # Inicializa estados
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat com o PowerRobot :robot_face:")

    # Sidebar para upload
    with st.sidebar:
        st.subheader("Seus documentos")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        if st.button("Confirmar") and pdf_docs:
            with st.spinner("Processando documentos..."):
                text = extract_text_from_pdfs(pdf_docs)
                text_chunks = split_text_into_chunks(text)
                vectorstore = create_vectorstore(text_chunks)
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
            # Força re-renderização (para manter no fim da tela)
            st.rerun()


if __name__ == '__main__':
    main()

