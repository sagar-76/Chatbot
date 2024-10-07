# Import necessary libraries
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


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    config={
        "max_length": 450,
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.eos_token_id
    }
)


from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=hf_pipeline)


prompt_template = """
You are an advanced AI assistant trained to provide insightful answers based on the given context.

Context: {context}

Question: {question}

Based on the context provided, please formulate a concise and informative answer, ensuring that all relevant information is included:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


st.title("Question Answering System with LangChain & GPT-2")


uploaded_file = st.file_uploader("Upload a document (PDF or TXT) for context", type=["pdf", "txt"])

if uploaded_file is not None:
    document_text = ""


    if uploaded_file.type == "application/pdf":

        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            document_text += page.extract_text() + "\n"
    else:

        stringio = BytesIO(uploaded_file.getvalue())
        document_text = stringio.read().decode("utf-8", errors='ignore')


    st.write("Document Content:")
    st.write(document_text)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(document_text)


    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)


    retriever = vectorstore.as_retriever()


    combine_documents_chain = load_qa_chain(llm, chain_type="stuff")


    qa_chain = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever
    )


    question = st.text_input("Ask a question based on the document")

    if st.button("Get Answer"):
        if question:

            retrieved_answer = qa_chain.run(question)
            st.write("Answer:")
            st.write(retrieved_answer)
        else:
            st.write("Please enter a question.")
