import streamlit as st
import os
import tempfile
import nest_asyncio
from dotenv import load_dotenv

from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Sidebar input for Hugging Face API Key
def get_hf_api_key():
    with st.sidebar:
        api_key = st.text_input(
            "🔑 Enter your Hugging Face API Key",
            type="password",
            help="Provide your personal Hugging Face API key to use models"
        )
    if not api_key:
        st.warning("Please enter your API key to proceed.")
        st.stop()
    return api_key

# Load and process PDF
def load_and_process_pdf(uploaded_file):
    try:
        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1️⃣ Load PDF
        status_text.text("📖 Loading PDF...")
        progress_bar.progress(20)
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        if not docs:
            raise Exception("No content found in PDF")
        st.success(f"✅ Successfully loaded {len(docs)} pages.")

        # 2️⃣ Split text
        status_text.text("✂️ Splitting text into chunks...")
        progress_bar.progress(40)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise Exception("No text chunks created from PDF")
        st.success(f"✅ Successfully split into {len(chunks)} chunks.")

        # 3️⃣ Create embeddings and vector store
        status_text.text("🧠 Creating embeddings...")
        progress_bar.progress(60)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=None)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        progress_bar.progress(100)
        status_text.text("✅ PDF ready for Q&A!")
        os.unlink(tmp_file_path)
        return retriever, True

    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None, False

# Format retrieved docs
def format_docs(r_docs):
    return "\n\n".join(doc.page_content for doc in r_docs)

# Create QA function using transformers pipeline
def answer_question(llm_pipeline, context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    output = llm_pipeline(prompt, max_length=512)[0]["generated_text"]
    return output

# Main app
def main():
    hf_api_key = get_hf_api_key()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

    st.title("📚 PDF Q&A Chatbot (Hugging Face)")
    st.markdown("Upload a PDF and ask questions about its content!")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size} bytes)")

        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                retriever, success = load_and_process_pdf(uploaded_file)
                if success:
                    st.session_state.retriever = retriever
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success("🎉 PDF processed successfully! You can now ask questions.")
                else:
                    st.error("Failed to process PDF. Please try another file.")

        if "retriever" in st.session_state:
            st.divider()
            st.subheader("💬 Ask Questions")

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Initialize LLM pipeline
            llm_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1
            )

            # Chat input
            if prompt := st.chat_input("Ask a question about the PDF..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Retrieve context and answer
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        docs = st.session_state.retriever.get_relevant_documents(prompt)
                        context_text = format_docs(docs)
                        response = answer_question(llm_pipeline, context_text, prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px; margin: 20px 0;">
            <h3>📤 Upload a PDF to get started</h3>
            <p>Select a PDF file using the file uploader above</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
