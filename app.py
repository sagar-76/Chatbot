import os
from langchain.schema import Document  # Updated import for Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import streamlit as st
import PyPDF2  # Import PyPDF2 at the top

# Set up Hugging Face API key (you can set it in your environment)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize the Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Function to upload documents
def upload_documents():
    uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf"], accept_multiple_files=True)
    documents = []
    for uploaded_file in uploaded_files:
        # Read the content of the file
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # Use PyPDF2 to extract text from PDF
            reader = PyPDF2.PdfReader(uploaded_file)  # No context manager
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""  # Handle None cases

        # Create a Document object
        documents.append(Document(page_content=content, metadata={"name": uploaded_file.name}))

    return documents


# Function to create the RAG application
def create_rag_app():
    st.title("RAG Application with LangChain and Hugging Face")

    # Upload documents
    documents = upload_documents()

    if documents:
        # Create a vector store
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Create a RetrievalQA chain
        # Correct initialization for HuggingFaceHub
        llm = HuggingFaceHub(repo_id="gpt2")  # Change to your preferred LLM (e.g., "gpt2" or another model)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

        # User input for question
        user_question = st.text_input("Ask a question based on the uploaded documents:")

        if user_question:
            answer = qa_chain.run(user_question)
            st.write("Answer:", answer)


# Run the application
if __name__ == "__main__":
    create_rag_app()
