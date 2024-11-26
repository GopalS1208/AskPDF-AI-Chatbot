from dotenv import load_dotenv
import streamlit as st
#from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import PyPDF2
import re
import os
from langchain.prompts import PromptTemplate


def generate_pdf_reader(pdf_docs):
    pdf_text=""
    for pdf in pdf_docs:
        pdf_reader=PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text+=page.extract_text()
    return pdf_text

def generate_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks

def generate_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def generate_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
    in provided context just respond as, "Answer is not available in the given PDF", dont provide wrong answer \n
    Context:\n {context}?\n
    Question:\n {question}?\n

    Answer:
    """

    model = OpenAI(temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(knowledge_base,user_question):
    docs = knowledge_base.similarity_search(query=user_question)
    chain = generate_conversational_chain()

    response = chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True)
    
    return response

def get_citations(knowledge_base,user_question):
    retriever = knowledge_base.as_retriever()
    citations = retriever.get_relevant_documents(user_question)
    return citations

def main():
    load_dotenv()
    #print(os.getenv('OPENAI_API_KEY'))
    st.set_page_config(page_title='Chat with your PDF')
    st.header('Ask your PDF:')

    user_question=st.chat_input("Please feel free to ask anything about the uploaded PDF:")

    with st.sidebar:
        st.title("MENU")
        #upload file
        pdf_docs=st.file_uploader('Upload Files',type='pdf', accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                st.success("Done")

    if user_question:
        raw_text=generate_pdf_reader(pdf_docs)
        text_chunks=generate_text_chunks(raw_text)
        knowledge_base=generate_embeddings(text_chunks)
        response=user_input(knowledge_base, user_question)
        citations=get_citations(knowledge_base,user_question)

        st.write("Query:", user_question)
        st.write("Reply:",response["output_text"])
        st.write("Citations:",citations)

if __name__=='__main__':
    main()