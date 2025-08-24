import streamlit as st
import os
import tempfile
import asyncio
import nest_asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables (for local development)
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def get_api_key():
    """Get API key from environment or Streamlit secrets"""
    try:
        # Try environment variable first (for local development)
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # If not found, try Streamlit secrets (for deployment)
        if not api_key:
            try:
                api_key = st.secrets["GOOGLE_API_KEY"]
            except (KeyError, AttributeError):
                pass
        
        if not api_key:
            st.error("🔑 Please set your GOOGLE_API_KEY!")
            st.info("For deployment, add it to Streamlit secrets. For local development, add it to your .env file.")
            st.stop()
            
        return api_key
    except Exception as e:
        st.error(f"Error getting API key: {e}")
        st.stop()

def load_and_process_pdf(uploaded_file, api_key):
    """Load and process the uploaded PDF"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Load PDF
        status_text.text("📖 Loading PDF...")
        progress_bar.progress(20)
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        if not docs:
            raise Exception("No content found in PDF")
        
        st.success(f"✅ Successfully loaded {len(docs)} pages.")
        
        # 2. Split into chunks
        status_text.text("✂️ Splitting text into chunks...")
        progress_bar.progress(40)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)
        
        if not chunks:
            raise Exception("No text chunks created from PDF")
            
        st.success(f"✅ Successfully split into {len(chunks)} chunks.")
        
        # 3. Create embeddings and vector store
        status_text.text("🧠 Creating embeddings...")
        progress_bar.progress(60)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        status_text.text("🗄️ Creating vector database...")
        progress_bar.progress(80)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=None  # Use in-memory storage for deployment
        )
        
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # 4. Create LLM
        status_text.text("🤖 Initializing AI model...")
        progress_bar.progress(90)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Ready to answer questions!")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return retriever, llm, True
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None, None, False

def format_docs(r_docs):
    """Format retrieved documents into context text"""
    context_text = "\n\n".join(doc.page_content for doc in r_docs)
    return context_text

def create_qa_chain(retriever, llm):
    """Create the question-answering chain"""
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions based on provided documents.

Instructions:
- Answer based ONLY on the information provided in the context below
- Be accurate and specific when the context supports it
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question"
- Provide relevant details and specifics when available
- Keep your response clear and well-organized

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create parallel chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    
    # Create parser
    parser = StrOutputParser()
    
    # Create main chain
    main_chain = parallel_chain | prompt | llm | parser
    
    return main_chain

# Main Streamlit App
def main():
    # Get API key
    api_key = get_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # App header
    st.title("📚 PDF Q&A Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Sidebar for instructions
    with st.sidebar:
        st.title("📋 How to Use")
        st.markdown("""
        1. **Upload PDF**: Click the upload button and select your PDF file
        2. **Wait for Processing**: The app will extract and process the text
        3. **Ask Questions**: Type your question in the input box
        4. **Get Answers**: The AI will answer based on the PDF content
        """)
        
        st.title("ℹ️ Features")
        st.markdown("""
        - **Smart Chunking**: Splits documents intelligently
        - **Semantic Search**: Finds most relevant sections
        - **Context-Aware**: Answers based only on PDF content
        - **Error Handling**: Robust processing pipeline
        """)
    
    st.warning("⚠️ **Note:** This app works best with text-based PDFs. Images and tables may not be processed accurately.")
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process PDF only once per session or when file changes
        if ("processed_file" not in st.session_state or 
            st.session_state.processed_file != uploaded_file.name):
            
            with st.spinner("Processing PDF..."):
                retriever, llm, success = load_and_process_pdf(uploaded_file, api_key)
                
                if success:
                    # Store components in session state
                    st.session_state.retriever = retriever
                    st.session_state.llm = llm
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.qa_chain = create_qa_chain(retriever, llm)
                    
                    st.balloons()  # Celebration effect
                    st.success("🎉 PDF processed successfully! You can now ask questions.")
                else:
                    st.error("Failed to process PDF. Please try again with a different file.")
        
        # Question-answering interface
        if "qa_chain" in st.session_state:
            st.divider()
            st.subheader("💬 Ask Questions")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the PDF..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.qa_chain.invoke(prompt)
                            st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    else:
        # Show upload prompt
        st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px; margin: 20px 0;">
            <h3>📤 Upload a PDF to get started</h3>
            <p>Select a PDF file using the file uploader above</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
