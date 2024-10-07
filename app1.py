# Import necessary libraries
import streamlit as st
import requests
from io import BytesIO
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain import PromptTemplate

# Set up Hugging Face API token
API_TOKEN = "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"

# Define the Hugging Face Inference API URL
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# Define headers for Hugging Face API
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query the Hugging Face model via API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}  # Return error message if the request fails

# Define a custom prompt template
prompt_template = """
You are a helpful assistant. Based on the following context, answer the question provided.

Context: {context}

Question: {question}

Answer in a detailed and informative manner:
"""

# Initialize Streamlit app
st.title("RAG System with FAISS, Hugging Face API, and Prompt Template")

# File uploader for document input (PDF or text)
uploaded_file = st.file_uploader("Upload a document (PDF or TXT) for context", type=["pdf", "txt"])

if uploaded_file is not None:
    document_text = ""

    # Handle the uploaded file (PDF or TXT)
    if uploaded_file.type == "application/pdf":
        # Read PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            document_text += page.extract_text() + "\n"
    else:
        # Read TXT file
        stringio = BytesIO(uploaded_file.getvalue())
        document_text = stringio.read().decode("utf-8", errors='ignore')

    # Display the uploaded document content
    st.write("Document Content:")
    st.write(document_text)

    # Split document into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(document_text)

    # Create Document objects for each chunk
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Use Hugging Face sentence transformer model to create embeddings for FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store for retrieval
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Retrieve documents using FAISS
    retriever = vectorstore.as_retriever()

    # Input field for the user's question
    question = st.text_input("Ask a question based on the document")

    if st.button("Get Answer"):
        if question:
            # Retrieve relevant documents from FAISS
            relevant_docs = retriever.get_relevant_documents(question)
            context = " ".join([doc.page_content for doc in relevant_docs])

            # Fill the prompt template with the context and question
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            formatted_prompt = prompt.format(context=context, question=question)

            # Query the Hugging Face model API with the formatted prompt
            response = query({"inputs": {"question": question, "context": context}})

            # Print the response for debugging purposes
            st.write("Response from API:")
            st.json(response)  # Display the whole response in JSON format

            # Check for error in the response
            if "error" in response:
                answer = response["error"]
            else:
                answer = response.get("answer", "Sorry, I couldn't find an answer.")

            # Display the prompt and answer
            st.write("Prompt:")
            st.write(formatted_prompt)

            st.write("Answer:")
            st.write(answer)
        else:
            st.write("Please enter a question.")
