import os
import io
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# PDF & Text Processing
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Multi-PDF ChatBot API", version="1.0.0")

# Enable CORS for Vercel and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chunks: List[str]
    embeddings: List[List[float]]
    model_name: Optional[str] = "gemini-1.5-flash"
    api_key: Optional[str] = None
    top_k: Optional[int] = 4


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Multi-PDF ChatBot Serverless API"}


@app.post("/api/process")
async def process_documents(
    files: List[UploadFile] = File(...),
    api_key: Optional[str] = Form(None)
):
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Google Gemini API Key is required. Please provide it in the sidebar or set GOOGLE_API_KEY."
        )

    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    raw_text = ""
    filenames = []

    for file in files:
        filenames.append(file.filename)
        try:
            content = await file.read()
            pdf_reader = PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Error reading PDF '{file.filename}': {str(e)}"
            )

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any readable text from the uploaded PDF documents."
        )

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(raw_text)

    if not chunks:
        raise HTTPException(status_code=422, detail="No text chunks generated.")

    # Generate embeddings using Google's text-embedding-004
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key
        )
        vectors = embeddings_model.embed_documents(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

    return {
        "success": True,
        "filenames": filenames,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "embeddings": vectors
    }


@app.post("/api/chat")
async def chat_with_docs(request: ChatRequest):
    key = request.api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Google Gemini API Key is required. Please provide it in the settings."
        )

    if not request.chunks or not request.embeddings:
        raise HTTPException(
            status_code=400,
            detail="No processed document chunks found. Please upload and process PDFs first."
        )

    # Embed user question
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key
        )
        query_vector = np.array(
            embeddings_model.embed_query(request.question),
            dtype=np.float32
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed question: {str(e)}"
        )

    # Compute cosine similarity across document chunks
    doc_vectors = np.array(request.embeddings, dtype=np.float32)
    norm_query = np.linalg.norm(query_vector)
    norm_docs = np.linalg.norm(doc_vectors, axis=1)
    similarities = np.dot(doc_vectors, query_vector) / (norm_docs * norm_query + 1e-10)

    # Retrieve Top-K most relevant chunks
    k = min(request.top_k or 4, len(request.chunks))
    top_indices = np.argsort(similarities)[::-1][:k]
    retrieved_chunks = [request.chunks[int(idx)] for idx in top_indices if int(idx) < len(request.chunks)]

    context_text = "\n\n".join(retrieved_chunks)

    # Generate answer with Google Gemini LCEL chain
    prompt_template = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
If the answer is not in the provided context, just say: "Answer is not available in the context." Do not provide inaccurate or hallucinated answers.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        model = ChatGoogleGenerativeAI(
            model=request.model_name or "gemini-1.5-flash",
            temperature=0.3,
            google_api_key=key
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = prompt | model | StrOutputParser()
        reply = chain.invoke({"context": context_text, "question": request.question})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini generation error: {str(e)}"
        )

    return {
        "reply": reply,
        "retrieved_chunks": retrieved_chunks
    }


# Fallback root route for local testing via uvicorn
@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multi-PDF ChatBot API"}
