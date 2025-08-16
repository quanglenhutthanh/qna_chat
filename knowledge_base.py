import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from datetime import datetime

def extract_text_from_file(uploaded_file):
    """Enhanced text extraction with better error handling."""
    text = ""
    
    try:
        if uploaded_file.type in ["text/plain", "text/markdown"]:
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Clean the text
        text = clean_text(text)
        return text.strip()
    except Exception as e:
        raise Exception(f"Error processing {uploaded_file.name}: {str(e)}")

def clean_text(text):
    """Clean and normalize text."""
    import re
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()


def split_text_into_chunks(text, chunk_size=800, chunk_overlap=100):
    """Enhanced text splitting with semantic boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators
    )
    return splitter.split_text(text)


def add_documents_to_vectorstore(uploaded_files, embedding_fn, vector_store=None):
    """Enhanced with deduplication and better metadata."""
    all_texts = []
    metadatas = []
    seen_chunks = set()  # For deduplication

    for uploaded_file in uploaded_files:
        text = extract_text_from_file(uploaded_file)
        
        if not text:
            return None, f"⚠️ No text found in {uploaded_file.name}, skipping..."

        chunks = split_text_into_chunks(text)
        
        for i, chunk in enumerate(chunks):
            # Simple deduplication
            chunk_hash = hash(chunk)
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                all_texts.append(chunk)
                metadatas.append({
                    "source": uploaded_file.name,
                    "chunk_id": i,
                    "file_size": len(text),
                    "upload_time": datetime.now().isoformat()
                })

    if all_texts:
        new_store = FAISS.from_texts(all_texts, embedding_fn, metadatas=metadatas)
        if vector_store is None:
            vector_store = new_store
        else:
            vector_store.merge_from(new_store)

    return vector_store, f"✅ Added {len(all_texts)} chunks from {len(uploaded_files)} files."


def view_knowledge_base(vector_store):
    """Return total vectors stored in the KB."""
    if vector_store:
        return f"Total vectors stored: {vector_store.index.ntotal}"
    else:
        return "📂 Knowledge base is empty."

def validate_file(uploaded_file):
    """Validate uploaded file."""
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        raise ValueError(f"File {uploaded_file.name} is too large (>10MB)")
    
    if uploaded_file.type not in ["text/plain", "text/markdown", "application/pdf"]:
        raise ValueError(f"Unsupported file type: {uploaded_file.type}")
    
    return True

def retrieve_with_reranking(vector_store, query, k=5, rerank_k=3):
    """Retrieve documents with optional reranking."""
    # Get more candidates initially
    docs = vector_store.similarity_search_with_score(query, k=k)
    
    # Simple reranking based on query overlap
    def calculate_overlap(doc, query):
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        return len(query_words.intersection(doc_words)) / len(query_words)
    
    # Rerank and return top results
    reranked = [(doc, score, calculate_overlap(doc, query)) for doc, score in docs]
    reranked.sort(key=lambda x: x[2], reverse=True)
    
    return [doc for doc, _, _ in reranked[:rerank_k]]

def get_kb_stats(vector_store):
    """Get knowledge base statistics."""
    if not vector_store:
        return {"status": "empty", "total_chunks": 0}
    
    return {
        "status": "active",
        "total_chunks": vector_store.index.ntotal,
        "dimensions": vector_store.index.d,
        "index_type": type(vector_store.index).__name__
    }
