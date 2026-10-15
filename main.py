import os
import shutil
import streamlit as st
from dotenv import load_dotenv

# Try importing modern pypdf, fallback to PyPDF2 for backwards compatibility
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# Text splitter compatible with modern langchain-text-splitters and legacy langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files safely."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading {getattr(pdf, 'name', 'PDF')}: {e}")
    return text


def get_text_chunks(text):
    """Split extracted text into overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key=None):
    """Generate embeddings and persist FAISS vector store locally."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(model_name="gemini-1.5-flash", api_key=None):
    """Create a question-answering chain using modern LCEL (LangChain Expression Language)."""
    prompt_template = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
If the answer is not in the provided context, just say: "Answer is not available in the context." Do not provide inaccurate or hallucinated answers.

Context:
{context}

Question:
{question}

Answer:"""

    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Modern LangChain Expression Language (LCEL) chain
    chain = prompt | model | StrOutputParser()
    return chain


def user_input(user_question, model_name="gemini-1.5-flash", api_key=None):
    """Process user query against the FAISS index and return Gemini response."""
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ No processed documents found. Please upload and process PDF files first in the sidebar.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        # allow_dangerous_deserialization is required for local FAISS pickle deserialization in modern LangChain
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)

        # Combine retrieved document chunks into context
        context_text = "\n\n".join([doc.page_content for doc in docs])

        chain = get_conversational_chain(model_name=model_name, api_key=api_key)

        response = chain.invoke(
            {"context": context_text, "question": user_question}
        )

        st.markdown("### 💬 Response")
        st.write(response)

        with st.expander("🔍 View Retrieved Document Chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                preview = doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
                st.caption(preview)
    except Exception as e:
        st.error(f"Error during response generation: {e}")


def main():
    st.set_page_config(
        page_title="Multi-PDF ChatBot | Google Gemini & FAISS",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Multi-PDF ChatBot with Google Gemini")
    st.markdown(
        "Upload multiple PDF documents and ask questions. "
        "Powered by **Google Gemini**, **FAISS Vector Store**, and **LangChain**."
    )

    # API Key retrieval (from env or sidebar input)
    env_api_key = os.getenv("GOOGLE_API_KEY", "")

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input(
            "Google API Key",
            value=env_api_key,
            type="password",
            help="Enter your Gemini API key from Google AI Studio (https://aistudio.google.com/)"
        )
        active_api_key = api_key_input if api_key_input else env_api_key

        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0,
            help="Select the Gemini model to use for generating answers."
        )

        st.divider()
        st.header("📁 Document Management")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process", use_container_width=True):
            if not active_api_key:
                st.error("Please provide a valid Google Gemini API Key above.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDF documents and creating vector index..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded PDF files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks, api_key=active_api_key)
                        st.success(f"Processed {len(pdf_docs)} file(s) into {len(text_chunks)} chunks!")

        if os.path.exists("faiss_index"):
            st.divider()
            if st.button("🗑️ Clear Vector Index", use_container_width=True):
                shutil.rmtree("faiss_index", ignore_errors=True)
                st.info("Vector index cleared.")
                st.rerun()

    # Main QA area
    user_question = st.text_input(
        "Ask a Question from the PDF Files:",
        placeholder="e.g. What are the key points covered in the document?"
    )

    if user_question:
        if not active_api_key:
            st.warning("⚠️ Please provide a Google Gemini API Key in the sidebar or via .env file to continue.")
        else:
            user_input(user_question, model_name=model_name, api_key=active_api_key)


if __name__ == "__main__":
    main()