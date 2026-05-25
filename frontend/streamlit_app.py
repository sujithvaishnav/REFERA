# app.py
import streamlit as st
import requests
import sseclient
import json
from datetime import datetime
import time

# Custom CSS for better visual appeal
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        margin: 10px 0;
        padding: 10px;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* User message styling */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:contains("user")) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:contains("assistant")) {
        background: #f0f2f6;
        border-left: 4px solid #764ba2;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #764ba2;
        border-radius: 10px;
        padding: 1rem;
        background: rgba(118, 75, 162, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #764ba2;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    
    /* Status indicator */
    .status-indicator {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 5px 10px;
        border-radius: 20px;
        background: #28a745;
        color: white;
        font-size: 12px;
        z-index: 999;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #764ba2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Source cards */
    .source-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #764ba2;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        
        .main {
            padding: 1rem;
        }
    }
</style>

<script>
    // Auto-scroll to bottom of chat
    function scrollToBottom() {
        const chatContainer = document.querySelector('.stChatMessage');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Call on load and after each message
    window.addEventListener('load', scrollToBottom);
    setInterval(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now()

# Title with animation
st.title("📚 ReferA")
st.markdown("### Your Intelligent Document Assistant")

# Sidebar configuration
with st.sidebar:
    st.markdown("## 📁 Document Management")
    
    # Status indicator
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        st.markdown('<div class="loading"></div>', unsafe_allow_html=True)
    with status_col2:
        st.markdown("**System Online**")
    
    st.markdown("---")
    
    # Fetch documents
    try:
        with st.spinner("Loading documents..."):
            docs_response = requests.get(
                "http://127.0.0.1:8000/documents",
                timeout=5
            )
            
            if docs_response.status_code == 200:
                available_docs = docs_response.json()["documents"]
                
                selected_docs = st.multiselect(
                    "📄 Select Documents to Query",
                    available_docs,
                    default=available_docs,
                    help="Choose which documents to search through"
                )
            else:
                st.error("⚠️ Could not fetch documents. Make sure the backend is running.")
                selected_docs = []
    except requests.exceptions.RequestException:
        st.error("⚠️ Backend server not reachable. Please check if FastAPI is running on port 8000.")
        selected_docs = []
    
    st.markdown("---")
    
    # File upload section
    st.markdown("## 📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload PDF files to add to your knowledge base"
    )
    
    if uploaded_file:
        files = {"file": uploaded_file}
        
        with st.spinner("Uploading and processing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/upload",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.success("✅ PDF uploaded and processed successfully!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Upload failed. Please try again.")
            except requests.exceptions.RequestException:
                st.error("❌ Could not connect to backend server.")
    
    st.markdown("---")
    
    # Additional info
    with st.expander("ℹ️ About"):
        st.markdown("""
        This AI assistant can:
        - Answer questions from your documents
        - Provide source citations
        - Handle multiple documents simultaneously
        
        **How to use:**
        1. Select documents from the sidebar
        2. Ask questions in the chat
        3. View sources under each response
        """)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong><br>
                            📄 <strong>File:</strong> {source['source']}<br>
                            📖 <strong>Page:</strong> {source['page']}<br>
                            📝 <strong>Snippet:</strong><br>
                            <em>{source['snippet'][:300]}{'...' if len(source['snippet']) > 300 else ''}</em>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
query = st.chat_input("💬 Ask a question about your documents...")

if query and selected_docs:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        try:
            response = requests.get(
                "http://127.0.0.1:8000/ask",
                params={
                    "query": query,
                    "selected_docs": ",".join(selected_docs)
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                client = sseclient.SSEClient(response)
                sources = []
                
                # Show loading animation
                response_container.markdown('<div class="loading"></div> Thinking...', unsafe_allow_html=True)
                
                for event in client.events():
                    data = json.loads(event.data)
                    
                    if "token" in data:
                        full_response += data["token"]
                        response_container.markdown(
                            full_response + '<span class="loading"></span>',
                            unsafe_allow_html=True
                        )
                    
                    if "done" in data:
                        response_container.markdown(full_response)
                        sources = data.get("sources", [])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
                # Display sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}:</strong><br>
                                📄 <strong>File:</strong> {source['source']}<br>
                                📖 <strong>Page:</strong> {source['page']}<br>
                                📝 <strong>Snippet:</strong><br>
                                <em>{source['snippet'][:300]}{'...' if len(source['snippet']) > 300 else ''}</em>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Backend error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")

elif query and not selected_docs:
    st.warning("⚠️ Please select at least one document to query.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by RAG Technology 🤖 | Responses are generated based on selected documents</p>",
    unsafe_allow_html=True
)