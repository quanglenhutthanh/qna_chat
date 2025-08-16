import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded TXT, MD, or PDF file."""
    text = ""

    if uploaded_file.type in ["text/plain", "text/markdown"]:
        text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


def split_text_into_chunks(text, chunk_size=800, chunk_overlap=100):
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def add_documents_to_vectorstore(uploaded_files, embedding_fn, vector_store=None):
    """Process uploaded files, split into chunks, and add to FAISS index."""
    all_texts = []
    metadatas = []

    for uploaded_file in uploaded_files:
        text = extract_text_from_file(uploaded_file)

        if not text:
            # Returning warning info for UI to handle
            return None, f"⚠️ No text found in {uploaded_file.name}, skipping..."

        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            all_texts.append(chunk)
            metadatas.append({"source": uploaded_file.name})

    if all_texts:
        new_store = FAISS.from_texts(all_texts, embedding_fn, metadatas=metadatas)
        if vector_store is None:
            vector_store = new_store
        else:
            vector_store.merge_from(new_store)

    return vector_store, "✅ Documents added to knowledge base."


def view_knowledge_base(vector_store):
    """Return total vectors stored in the KB."""
    if vector_store:
        return f"Total vectors stored: {vector_store.index.ntotal}"
    else:
        return "📂 Knowledge base is empty."
