import os
import sys
import json
import time
from datetime import datetime
import tempfile

# Force pure-Python implementation of protobuf to prevent OpenTelemetry/Protobuf crashes on Python 3.12+
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

# Add backend directory to sys.path so RAG pipeline modules can be imported directly
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import importlib
import rag.parser
import rag.chunker
import rag.embeddings
import rag.vectordb
import rag.retriever
import rag.generator
import rag.summarizer
import rag.reranker
import rag.cache

importlib.reload(rag.parser)
importlib.reload(rag.chunker)
importlib.reload(rag.embeddings)
importlib.reload(rag.vectordb)
importlib.reload(rag.retriever)
importlib.reload(rag.generator)
importlib.reload(rag.summarizer)
importlib.reload(rag.reranker)
importlib.reload(rag.cache)

try:
    from rag.parser import extract_text_from_pdf
    from rag.chunker import chunk_text
    from rag.embeddings import generate_embedding
    from rag.vectordb import store_chunks, delete_document_by_id, supabase
    from rag.retriever import hybrid_retrieve, build_bm25_index
    from rag.generator import generate_answer
    from rag.reranker import rerank_documents
except Exception as init_err:
    st.error(f"⚠️ Error initializing backend RAG modules: {init_err}")
    st.info("Please make sure you have filled in your .env file with valid SUPABASE_URL, SUPABASE_KEY, and GROQ_API_KEY.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="ReferA | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Academic & Research UI
st.markdown("""
<style>
    /* Global font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Hero Header */
    .hero-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .summary-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .summary-card h3 {
        color: #818cf8;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 10px;
    }

    .source-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
        border-left: 4px solid #38bdf8;
        color: #cbd5e1;
        font-size: 0.92rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    
    .source-header {
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 4px;
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 14px;
        margin: 10px 0;
        padding: 16px;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.25rem;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
    }
    
    .delete-btn > button {
        background: #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "user" not in st.session_state:
    st.session_state.user = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_session_id" not in st.session_state:
    st.session_state.loaded_session_id = None
if "last_uploaded_summary" not in st.session_state:
    st.session_state.last_uploaded_summary = None

# ==============================================================================
# AUTHENTICATION SCREEN
# ==============================================================================
if not st.session_state.user:
    st.markdown('<div class="hero-title">📚 ReferA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Production-Grade Multi-Document AI Research Assistant</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with tab_login:
            with st.form("login_form"):
                st.subheader("Welcome Back")
                email = st.text_input("Email Address", placeholder="researcher@university.edu")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                login_submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if login_submit:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    else:
                        with st.spinner("Authenticating..."):
                            try:
                                auth_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                                if auth_res.user:
                                    st.session_state.user = auth_res.user
                                    if hasattr(auth_res, "session") and auth_res.session and hasattr(auth_res.session, "access_token"):
                                        st.session_state.access_token = auth_res.session.access_token
                                        try:
                                            supabase.postgrest.auth(auth_res.session.access_token)
                                        except Exception:
                                            pass
                                    build_bm25_index(auth_res.user.id)
                                    st.success("Successfully authenticated! Loading workspace...")
                                    time.sleep(0.4)
                                    st.rerun()
                                else:
                                    st.error("Authentication failed. Please verify your credentials.")
                            except Exception as e:
                                err_msg = str(e)
                                if "getaddrinfo failed" in err_msg or "ConnectError" in err_msg or "11001" in err_msg:
                                    st.error("❌ **Supabase Connection Error**")
                                    st.warning(
                                        "Could not connect to your Supabase project. "
                                        "This usually happens if your Supabase project is **paused** or if the `SUPABASE_URL` in `.env` is invalid.\n\n"
                                        "**To fix this:**\n"
                                        "1. Open [supabase.com/dashboard](https://supabase.com/dashboard) and unpause your project (or create a new free project).\n"
                                        "2. Copy the **Project URL** & **anon key** into your `.env` file.\n"
                                        "3. Run `supabase_schema.sql` in the Supabase SQL Editor."
                                    )
                                elif "Invalid login credentials" in err_msg:
                                    st.error("❌ **Invalid Email or Password**")
                                    st.info("💡 If you haven't registered an account yet, please switch to the **'Create Account'** tab to register first.")
                                elif "Email not confirmed" in err_msg:
                                    st.error("❌ **Email Not Confirmed**")
                                    st.info("Please check your email inbox to confirm your account, or disable *'Confirm email'* in Supabase Dashboard > Authentication > Providers > Email.")
                                else:
                                    st.error(f"Login failed: {err_msg}")
                                
        with tab_signup:
            with st.form("signup_form"):
                st.subheader("New Researcher Registration")
                new_email = st.text_input("Email Address", placeholder="researcher@university.edu")
                new_password = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
                signup_submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_submit:
                    if not new_email or not new_password:
                        st.error("Please fill in all fields.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        with st.spinner("Creating your research workspace..."):
                            try:
                                sign_res = supabase.auth.sign_up({"email": new_email, "password": new_password})
                                st.success("✅ Account created successfully! You can now switch to the 'Sign In' tab to log in.")
                                if sign_res.user and not sign_res.session:
                                    st.info("ℹ️ If email confirmation is enabled in your Supabase project, please check your inbox to confirm your address before signing in.")
                            except Exception as e:
                                err_msg = str(e)
                                if "getaddrinfo failed" in err_msg or "ConnectError" in err_msg or "11001" in err_msg:
                                    st.error("❌ **Supabase Connection Error**")
                                    st.warning("Could not reach Supabase. Please verify that your Supabase project is active and unpaused.")
                                else:
                                    st.error(f"Registration failed: {err_msg}")
                                
    st.stop()

# ==============================================================================
# MAIN WORKSPACE (AUTHENTICATED USER)
# ==============================================================================
user = st.session_state.user
if not user:
    st.stop()

# Ensure PostgREST client is authenticated with user's JWT
if "access_token" in st.session_state and st.session_state.access_token:
    try:
        supabase.postgrest.auth(st.session_state.access_token)
    except Exception:
        pass
if not st.session_state.active_session_id:
    try:
        session_res = supabase.table("chat_sessions") \
            .select("id") \
            .eq("user_id", user.id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
            
        if session_res.data:
            st.session_state.active_session_id = session_res.data[0]["id"]
        else:
            new_title = f"Research Chat {datetime.now().strftime('%b %d, %H:%M')}"
            new_session_res = supabase.table("chat_sessions").insert({
                "user_id": user.id,
                "title": new_title
            }).execute()
            if new_session_res.data:
                st.session_state.active_session_id = new_session_res.data[0]["id"]
    except Exception as e:
        st.error(f"Failed to initialize chat session: {str(e)}")

# Load message history when session changes
if st.session_state.active_session_id and st.session_state.loaded_session_id != st.session_state.active_session_id:
    try:
        history_res = supabase.table("chat_messages") \
            .select("role", "content", "sources") \
            .eq("session_id", st.session_state.active_session_id) \
            .order("created_at", desc=False) \
            .execute()
            
        st.session_state.messages = []
        if history_res.data:
            for msg in history_res.data:
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "sources": msg.get("sources")
                })
        st.session_state.loaded_session_id = st.session_state.active_session_id
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")

# ==============================================================================
# SIDEBAR: DOCUMENT & SESSION MANAGEMENT
# ==============================================================================
with st.sidebar:
    st.markdown("### 👤 Researcher Profile")
    st.caption(f"Logged in as: **{user.email}**")
    
    if st.button("🚪 Sign Out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.user = None
        st.session_state.active_session_id = None
        st.session_state.messages = []
        st.session_state.loaded_session_id = None
        st.session_state.last_uploaded_summary = None
        st.rerun()
        
    st.markdown("---")
    
    # --- CHAT SESSION MANAGEMENT ---
    st.markdown("### 💬 Chat Sessions")
    try:
        sessions_res = supabase.table("chat_sessions") \
            .select("id", "title", "created_at") \
            .eq("user_id", user.id) \
            .order("created_at", desc=True) \
            .execute()
            
        sessions_list = sessions_res.data or []
        
        if sessions_list:
            session_dict = {f"{s['title']} ({s['id'][:6]})": s["id"] for s in sessions_list}
            session_labels = list(session_dict.keys())
            
            # Find current index
            current_idx = 0
            for idx, label in enumerate(session_labels):
                if session_dict[label] == st.session_state.active_session_id:
                    current_idx = idx
                    break
                    
            selected_label = st.selectbox(
                "Active Conversation",
                options=session_labels,
                index=current_idx
            )
            
            chosen_id = session_dict[selected_label]
            if chosen_id != st.session_state.active_session_id:
                st.session_state.active_session_id = chosen_id
                st.rerun()
        else:
            st.caption("No conversations yet.")
            
    except Exception as e:
        st.error(f"Error loading sessions: {str(e)}")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True):
            try:
                new_title = f"Research Chat {datetime.now().strftime('%b %d, %H:%M')}"
                new_session = supabase.table("chat_sessions").insert({
                    "user_id": user.id,
                    "title": new_title
                }).execute()
                if new_session.data:
                    st.session_state.active_session_id = new_session.data[0]["id"]
                    st.session_state.messages = []
                    st.session_state.loaded_session_id = st.session_state.active_session_id
                    st.rerun()
            except Exception as e:
                st.error(f"Error creating chat: {str(e)}")
                
    with col_btn2:
        if st.session_state.active_session_id:
            if st.button("🗑️ Delete Chat", use_container_width=True):
                try:
                    supabase.table("chat_sessions").delete().eq("id", st.session_state.active_session_id).execute()
                    st.session_state.active_session_id = None
                    st.session_state.messages = []
                    st.session_state.loaded_session_id = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting chat: {str(e)}")

    st.markdown("---")
    
    # --- DOCUMENT KNOWLEDGE BASE ---
    st.markdown("### 📚 Knowledge Base")
    available_docs = []
    doc_lookup = {}
    
    try:
        docs_res = supabase.table("documents") \
            .select("id", "filename", "summary") \
            .eq("user_id", user.id) \
            .order("created_at", desc=True) \
            .execute()
            
        docs_data = docs_res.data or []
        available_docs = [d["filename"] for d in docs_data]
        doc_lookup = {d["filename"]: d for d in docs_data}
        
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

    if available_docs:
        selected_docs = st.multiselect(
            "Filter Active Documents",
            options=available_docs,
            default=available_docs,
            help="Select one or more research papers to constrain your query."
        )
    else:
        st.info("Upload your first research paper below to start asking questions.")
        selected_docs = []

    st.markdown("---")
    
    # --- DOCUMENT UPLOAD ---
    st.markdown("### 📤 Ingest Research Papers")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload PDF papers to automatically parse, chunk, embed, and summarize."
    )
    
    if uploaded_file:
        upload_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
        if st.button(f"📥 Process {uploaded_file.name}", use_container_width=True):
            with st.spinner("Extracting text, embedding chunks & generating summary..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        pages = extract_text_from_pdf(tmp_path)
                        chunks = chunk_text(pages)
                        
                        summary_text = store_chunks(
                            chunks=chunks,
                            filename=uploaded_file.name,
                            generate_embedding=generate_embedding,
                            user_id=user.id
                        )

                        st.session_state.last_uploaded_summary = {
                            "filename": uploaded_file.name,
                            "summary": summary_text,
                            "pages": len(pages),
                            "chunks": len(chunks)
                        }

                        build_bm25_index(user.id)
                        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages!")
                        time.sleep(1.0)
                        st.rerun()
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                except Exception as e:
                    err_text = str(e)
                    if "row-level security" in err_text or "42501" in err_text:
                        st.error("❌ **Supabase Row-Level Security (RLS) Policy Error**")
                        st.warning(
                            "Your Supabase database requires updated RLS policies.\n\n"
                            "**To fix this instantly:**\n"
                            "1. Open your **Supabase Dashboard > SQL Editor**.\n"
                            "2. Run the updated `supabase_schema.sql` script (or run `ALTER TABLE documents DISABLE ROW LEVEL SECURITY;`)."
                        )
                    else:
                        st.error(f"Upload failed: {err_text}")

    # --- DOCUMENT VIEWER & DELETION ---
    if doc_lookup:
        with st.expander("📑 Manage Uploaded Papers"):
            for fname, doc_info in doc_lookup.items():
                st.markdown(f"**📄 {fname}**")
                if doc_info.get("summary"):
                    st.caption(f"{doc_info['summary'][:160]}...")
                if st.button(f"🗑️ Delete {fname}", key=f"del_{doc_info['id']}"):
                    delete_document_by_id(doc_info["id"], user.id)
                    build_bm25_index(user.id)
                    st.success(f"Deleted {fname}")
                    time.sleep(0.5)
                    st.rerun()
                st.markdown("---")

# ==============================================================================
# MAIN VIEWPORT: CONVERSATION & INTELLIGENCE
# ==============================================================================
st.markdown('<div class="hero-title">📚 ReferA</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent PDF Research Assistant with Hybrid Dense + Sparse Retrieval & Citations</div>', unsafe_allow_html=True)

# Prominently display executive summary of the newly ingested paper
if st.session_state.last_uploaded_summary:
    s_info = st.session_state.last_uploaded_summary
    st.markdown(f"""
    <div class="summary-card">
        <h3>✨ Executive Summary: <em>{s_info['filename']}</em></h3>
        <p>{s_info['summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Dismiss Summary & Continue Querying", use_container_width=True):
        st.session_state.last_uploaded_summary = None
        st.rerun()

# Display Chat History
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("sources"):
                with st.expander("🔍 Verified Document Citations"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">Citation [{idx}] — 📄 {src.get('source', 'Unknown')} (Page {src.get('page', 'N/A')})</div>
                            <em>"{src.get('snippet', '')}"</em>
                        </div>
                        """, unsafe_allow_html=True)

# User Query Input
user_query = st.chat_input("💬 Ask a technical question about your uploaded documents...")

if user_query:
    if not selected_docs:
        st.warning("⚠️ Please upload and select at least one document from the sidebar to query.")
    else:
        # 1. Render User Message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Persist User Message to Supabase
        try:
            supabase.table("chat_messages").insert({
                "session_id": st.session_state.active_session_id,
                "user_id": user.id,
                "role": "user",
                "content": user_query
            }).execute()
        except Exception as e:
            st.warning(f"Failed to record message in cloud: {str(e)}")

        # 3. Process RAG Pipeline with Streaming Output
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            try:
                # Construct conversation context from previous turns
                conversation_context = ""
                history_pairs = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history_pairs.append({"question": msg["content"], "answer": ""})
                    elif msg["role"] == "assistant" and history_pairs:
                        history_pairs[-1]["answer"] = msg["content"]
                        
                for turn in history_pairs[-5:]:
                    conversation_context += f"\nUser: {turn['question']}\nAssistant: {turn['answer']}\n"
                    
                enhanced_query = f"""
                Previous Conversation:
                {conversation_context}

                Current Question:
                {user_query}
                """
                
                # Hybrid Dense (pgvector) + Sparse (BM25) Retrieval via Reciprocal Rank Fusion
                retrieved_docs = hybrid_retrieve(
                    query=enhanced_query,
                    user_id=user.id,
                    selected_docs=selected_docs
                )
                
                # Cross-Encoder Reranking
                reranked_docs = rerank_documents(
                    query=enhanced_query,
                    retrieved_docs=retrieved_docs,
                    top_k=5
                )
                
                # LLM Generation via Groq LLaMA 3.1
                completion, sources = generate_answer(
                    query=enhanced_query,
                    retrieved_docs=reranked_docs
                )
                
                # Stream Tokens Live to UI
                def token_stream():
                    for chunk in completion:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                            
                streamed_answer = response_container.write_stream(token_stream())
                
                # 4. Record Assistant Response in State & Cloud
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": streamed_answer,
                    "sources": sources
                })
                
                try:
                    supabase.table("chat_messages").insert({
                        "session_id": st.session_state.active_session_id,
                        "user_id": user.id,
                        "role": "assistant",
                        "content": streamed_answer,
                        "sources": sources
                    }).execute()
                except Exception as e:
                    st.warning(f"Failed to record response in cloud: {str(e)}")

                # Display Citations
                if sources:
                    with st.expander("🔍 Verified Document Citations"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">Citation [{idx}] — 📄 {src.get('source', 'Unknown')} (Page {src.get('page', 'N/A')})</div>
                                <em>"{src.get('snippet', '')}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                            
            except Exception as e:
                response_container.error(f"❌ Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>ReferA v2.0 • Hybrid RAG Engine (pgvector + BM25 RRF + Cross-Encoder) • Powered by Groq & Supabase</p>",
    unsafe_allow_html=True
)
