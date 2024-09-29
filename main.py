import streamlit as st
from pypdf import PdfReader
from io import BytesIO
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the context, make sure to produce all details,
    if the answer is not present in the provided context, then just say "Particular Data Not Available",
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3)
    prompt= PromptTemplate(template=prompt_template,input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type='stuff',prompt=prompt)
    
    return chain

def user_input(user_ques):

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_ques)
    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents":docs,
            "question": user_ques
        }
        ,return_only_outputs=True
    )
    
    print(response)
    st.write("Reply: ",response['output_text'])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Preprocessing..."):
                    raw_text = get_pdf(pdf_docs)
                    chunks = get_text_chunks(raw_text) 
                    get_vector_store(chunks)  
                    st.success("PDF processing completed!")
            else:
                st.warning("Please upload at least one PDF file.")

    if st.sidebar.button('Clear Chat History'):
        clear_chat_history()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(placeholder="Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt) 
                full_response = ''.join(response['output_text']) 
                st.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
