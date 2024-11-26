from dotenv import load_dotenv
import streamlit as st
import PyPDF2 
from llama_index.core import Document, GPTVectorStoreIndex
from pathlib import Path
from llama_index.core.node_parser import SimpleNodeParser
import os
import openai

def generate_pdf_reader(uploaded_files):
    documents=[]
    pdf_text=""
    for pdf in uploaded_files:
        pdf_reader=PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text+=page.extract_text()
        document = Document(text=pdf_text, metadata={"file_name": pdf.name})
        documents.append(document)
    return documents

def generate_query_index(documents):
    # Initialize parser with markdown output (alternative: text)
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    query_index = GPTVectorStoreIndex(nodes)
    query_engine = query_index.as_query_engine()
    return query_engine

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with your PDF')
    st.header('Ask your PDF:')

    with st.sidebar:
        st.title("MENU")
        #upload file
        uploaded_files=st.file_uploader('Upload Files',type='pdf', accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                st.success("Files uploaded Successfully")

    user_question=st.chat_input("Please feel free to ask anything about the uploaded PDF:")

    if user_question:
        documents=generate_pdf_reader(uploaded_files)
        query_engine=generate_query_index(documents)
        response=query_engine.query(user_question)
        st.write("Query:", user_question)
        st.write("Response:", response.response)

if __name__=='__main__':
    main()