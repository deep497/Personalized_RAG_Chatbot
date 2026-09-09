from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import pdfplumber  # third party library to extract text
from dotenv import load_dotenv
import os

# Load API key: st.secrets works on Streamlit Cloud; .env works locally
load_dotenv()
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Add it to `.env` (local) or Streamlit Secrets (Cloud).")
    st.stop()


st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("upload a PDF file and start asking questions",type="pdf")

# Extract contents from the PDF and chunk it

if file is not None:
    # extract text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    #st.write(text)

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], 
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
    )
    #store embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    #get user question

    user_question = st.text_input("Type your Question here")



    #generate answer
    # question -> embeddings -> similarity search -> results to LLM -> response (CHAIN)
    
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    # define the LLM and prompt
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=1000,
        google_api_key=GEMINI_API_KEY
    )

    #provide the prompts
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful document analysis assistant.

Answer the user's question using the provided context from the uploaded PDF.

Give a detailed and comprehensive answer.

For analysis questions:
- Identify all important facts and figures available in the context.
- Include important numbers, dates, percentages, financial figures, names, and comparisons.
- Organize the answer using headings and bullet points.
- Do not unnecessarily shorten the answer.
- If the requested information is not present in the context, clearly say so.
- Do not invent information.

Context:
{context}"""
    ),
    (
        "user",
        "Question: {question}"
    )
])


    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if user_question:
        response = chain.invoke(user_question)
        st.write(response)