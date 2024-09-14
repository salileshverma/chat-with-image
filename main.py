import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize Tesseract path if required
# Update the path to the Tesseract executable if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(image):
    # Convert to grayscale for better OCR performance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    return text

def setup_vectorstore(text):
    embeddings = HuggingFaceEmbeddings()  # Ensure this is available in your environment
    # Split text into chunks for better retrieval
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_text(text)  # Changed to split_text for the latest API
    vectorstore = FAISS.from_texts(doc_chunks, embeddings)  # Changed to from_texts
    return vectorstore

def create_chain(vectorstore):
    llm = OpenAI(model="text-davinci-003", temperature=0)  # Adjust LLM based on availability and need
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit page configuration
st.set_page_config(
    page_title="Chat with Image",
    page_icon="🖼️",
    layout="centered"
)

st.title("🦙 Chat with your Image")

# Initializing the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_image = st.file_uploader(label="Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text using OCR
    extracted_text = load_image(image_np)

    if extracted_text:
        st.write("Extracted Text:")
        st.write(extracted_text)

        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(extracted_text)

        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask llama....")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
