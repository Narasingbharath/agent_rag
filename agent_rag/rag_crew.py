import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings  # AWS Titan embeddings
from crew import rag_crew  # Make sure this imports your CrewAI setup (agent + task)

# -------------------------
# Load PDF and extract text
# -------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---------------------------------------
# Create FAISS vector database with embeddings
# ---------------------------------------
def create_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    db = FAISS.from_documents(docs, embeddings)
    return db

# ---------------------------------------
# Answer a question using CrewAI & PDF
# ---------------------------------------
def answer_question(db, query):
    docs = db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    if not context.strip():
        return "I don’t know the answer."

    # Use CrewAI agent
    inputs = {"topic": f"Context:\n{context}\n\nQuestion: {query}"}
    result = rag_crew.kickoff(inputs=inputs)
    return str(result)

# -------------------------
# Streamlit UI
# -------------------------
st.title("RAG PDF Search Crew (Bedrock LLM)")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    st.info("Extracting PDF content and building vector database...")
    text = load_pdf(pdf_file)
    db = create_vector_db(text)
    st.success("PDF loaded and vector DB created!")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Fetching answer..."):
            answer = answer_question(db, query)
        st.write("### Answer:")
        st.write(answer)
