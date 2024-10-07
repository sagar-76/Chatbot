import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import PyPDF2

# Set up the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    config={
        "max_length": 450,  # Increase this to accommodate longer input sequences
        "max_new_tokens": 100,  # Set this based on how long the output should be
        "pad_token_id": tokenizer.eos_token_id  # Avoids the warning about pad_token_id
    }
)

# LangChain's Hugging Face integration
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define a custom prompt template for question answering
prompt_template = """
You are a knowledgeable assistant that provides detailed and accurate answers. 
Please use the given context to respond concisely and effectively.

Context: {context}

Question: {question}

Answer in a precise and clear manner:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize Streamlit app
st.title("Question Answering System with LangChain & GPT-2")

# File uploader for document input (PDF or text)
uploaded_file = st.file_uploader("Upload a document (PDF or TXT) for context", type=["pdf", "txt"])

if uploaded_file is not None:
    document_text = ""

    # Read the uploaded document based on its type
    if uploaded_file.type == "application/pdf":
        # Handle PDF files
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            document_text += page.extract_text() + "\n"
    else:
        # Handle TXT files
        stringio = BytesIO(uploaded_file.getvalue())
        document_text = stringio.read().decode("utf-8", errors='ignore')

    # Display the uploaded document
    st.write("Document Content:")
    st.write(document_text)

    # Split the document text into smaller chunks using a recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Define your chunk size here
        chunk_overlap=50  # Define how much overlap you want between chunks
    )
    chunks = text_splitter.split_text(document_text)

    # Create LangChain Document objects from the chunks
    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    # Create embeddings for document retrieval
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create a retriever using the vectorstore
    retriever = vectorstore.as_retriever()

    # Create the combine_documents_chain (this is required)
    combine_documents_chain = load_qa_chain(llm, chain_type="stuff")

    # Create the retrieval-based QA chain
    qa_chain = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever
    )

    # Input field for the user's question
    question = st.text_input("Ask a question based on the document")

    if st.button("Get Answer"):
        if question:
            # Retrieve relevant context and generate an answer
            context_docs = retriever.get_relevant_documents(question)

            # Combine relevant contexts into a single string (limited to fit GPT-2’s token limit)
            context = " ".join([doc.page_content for doc in context_docs[:3]])  # Limit to top 3 chunks
            formatted_prompt = prompt.format(context=context, question=question)

            # Log the prompt being passed to the model for debugging
            st.write(f"Generated Prompt:\n{formatted_prompt}")

            # Use the retriever and LLM to get the answer
            retrieved_answer = qa_chain.run(question)

            st.write("Answer:")
            st.write(retrieved_answer)
        else:
            st.write("Please enter a question.")
