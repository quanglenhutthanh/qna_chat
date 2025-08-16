import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from knowledge_base import (
    add_documents_to_vectorstore, 
    view_knowledge_base, 
    validate_file,
    retrieve_with_reranking,
    get_kb_stats
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ------------------------
# Load secrets
# ------------------------
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Add it to .streamlit/secrets.toml or .env")
    st.stop()

# ------------------------
# OpenAI & Embeddings setup
# ------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
embedding_fn = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# ------------------------
# Streamlit App Layout
# ------------------------
st.set_page_config(page_title="RAG with Enhanced Knowledge Base", layout="wide")
st.title("📚 Enhanced RAG App — Knowledge Base & Chat")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "⚙️ BC Prompt Generator", "📄 Knowledge Base"])

# ------------------------
# Tab 1: Chat
# ------------------------
with tab1:
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

        # Retrieve from FAISS using enhanced retrieval
        retrieved_context = ""
        vector_store = st.session_state.get("vector_store")
        if vector_store:
            try:
                # Use the new reranking function
                docs = retrieve_with_reranking(vector_store, user_input, k=5, rerank_k=3)
                retrieved_context = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                st.error(f"Error retrieving documents: {str(e)}")
                retrieved_context = ""

        # Build prompt
        conversation = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
        prompt = f"Context:\n{retrieved_context}\n\nConversation:\n{conversation}\n\nAnswer the last question."

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
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
                            
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    answer = "Sorry, I encountered an error while processing your request."

        st.session_state.chat_history.append(("assistant", answer))

# ------------------------
# Tab 2: BC Prompt Generator
# ------------------------
with tab2:
    st.subheader("⚙️ Dynamics 365 Business Central Prompt Generator")

    user_desc = st.text_area(
        "Enter your request (English or Vietnamese):",
        placeholder="Example: add field to table sales header"
    )

    if st.button("Generate Prompt"):
        if user_desc.strip():
            try:
                # 1. Detect & translate Vietnamese → English if needed
                translation_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "If the input is in Vietnamese, translate it into English. "
                                "If it's already in English, return it unchanged."
                            )
                        },
                        {"role": "user", "content": user_desc.strip()}
                    ]
                )
                translated_text = translation_response.choices[0].message.content.strip()

                # 2. Dynamically ask AI to generate a BC development prompt
                bc_prompt_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Microsoft Dynamics 365 Business Central AL developer. "
                                "Create a clear and detailed coding task prompt based on the given request, "
                                "so another AI can generate AL code for it. "
                                "Avoid generic phrasing, tailor the prompt to the request, and include best practices if relevant."
                            )
                        },
                        {"role": "user", "content": translated_text}
                    ]
                )
                generated_prompt = bc_prompt_response.choices[0].message.content.strip()

                st.subheader("Generated Prompt:")
                st.code(generated_prompt, language="markdown")
                
            except Exception as e:
                st.error(f"Error generating prompt: {str(e)}")
        else:
            st.warning("Please enter a description.")

# ------------------------
# Tab 3: Knowledge Base
# ------------------------

# Initialize session state variable
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with tab3:
    st.subheader("📄 Enhanced Knowledge Base Management")
    
    # Display current KB stats
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 View Knowledge Base Stats"):
            stats = get_kb_stats(st.session_state.get("vector_store"))
            st.json(stats)
    
    with col2:
        if st.button("🗑️ Clear Knowledge Base"):
            st.session_state.vector_store = None
            st.success("Knowledge base cleared!")
            st.rerun()
    
    # File upload section
    st.subheader("Upload & Embed Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT, MD, or PDF files (max 10MB each)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Validate files before processing
        valid_files = []
        for uploaded_file in uploaded_files:
            try:
                validate_file(uploaded_file)
                valid_files.append(uploaded_file)
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                continue
        
        if valid_files:
            try:
                st.session_state.vector_store, message = add_documents_to_vectorstore(
                    valid_files,
                    embedding_fn,
                    st.session_state.vector_store
                )
                st.success(message)
                
                # Show updated stats
                stats = get_kb_stats(st.session_state.vector_store)
                st.info(f"📈 Updated stats: {stats['total_chunks']} chunks stored")
                
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")

    # Display current knowledge base info
    st.subheader("Current Knowledge Base Status")
    if st.session_state.get("vector_store"):
        kb_info = view_knowledge_base(st.session_state.vector_store)
        st.info(kb_info)
        
        # Show detailed stats
        stats = get_kb_stats(st.session_state.vector_store)
        st.json(stats)
    else:
        st.warning("📂 Knowledge base is empty. Please upload and embed documents first.")





