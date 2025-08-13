import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------
# Load secrets
# ------------------------
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Add it to .streamlit/secrets.toml or .env")
    st.stop()

# ------------------------
# OpenAI & ChromaDB setup
# ------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="app_kb",
    embedding_function=embedding_fn
)

# ------------------------
# Streamlit App Layout
# ------------------------
st.set_page_config(page_title="RAG with ChromaDB", layout="wide")
st.title("📚 RAG App — Knowledge Base & Chat")

tab1, tab2 = st.tabs(["📄 Knowledge Base", "💬 Chat"])

# ------------------------
# Tab 1: Knowledge Base
# ------------------------
with tab1:
    st.subheader("Upload & Embed Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT, MD, or PDF files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = ""

            # Handle TXT and MD
            if uploaded_file.type in ["text/plain", "text/markdown"]:
                text = uploaded_file.read().decode("utf-8")

            # Handle PDF
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Skip if no text extracted
            if not text.strip():
                st.warning(f"⚠️ No text found in {uploaded_file.name}, skipping...")
                continue

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_text(text)

            # Add chunks to collection
            for i, chunk in enumerate(chunks):
                doc_id = f"{uploaded_file.name}-chunk-{i}"
                collection.add(
                    documents=[chunk],
                    ids=[doc_id],
                    metadatas={"source": uploaded_file.name}
                )

        st.success("✅ Documents added to knowledge base.")

    if st.button("View Current Knowledge Base"):
        st.write(f"Total documents stored: {collection.count()}")

# ------------------------
# Tab 2: Chat
# ------------------------
import streamlit as st

with tab2:
    st.subheader("💬 Chat with Your Knowledge Base")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for role, content in st.session_state.chat_history:
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

    # Add sticky chat input
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 1rem;
            width: 80%;
            background-color: white;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Chat input (always at bottom)
    if user_input := st.chat_input("Type your question..."):
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Retrieve from KB
        results = collection.query(query_texts=[user_input], n_results=3)
        retrieved_context = "\n".join(results["documents"][0]) if results["documents"] else ""

        # Build prompt
        conversation = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
        prompt = f"Context:\n{retrieved_context}\n\nConversation:\n{conversation}\n\nAnswer the last question."

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                 # Optional: collapsible context view
                if retrieved_context:
                    with st.expander("🔍 View retrieved context"):
                        st.markdown(retrieved_context)

        st.session_state.chat_history.append(("assistant", answer))


