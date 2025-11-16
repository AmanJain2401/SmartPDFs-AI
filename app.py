#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\activate
import streamlit as st
from dotenv import load_dotenv # Used to use variables inside .env ie my tokens
from PyPDF2 import PdfReader

from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter # Used to divide text chunks
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS #Stores all the embeddings (numericals) & stores Locally:
from langchain.memory import ConversationBufferMemory # Keeps the memory of prev questions
from langchain.chains import ConversationalRetrievalChain # Allows to chat with context with vectorstore and have some memory to it

from Templates import css, bot_template, user_template

def get_pdf_text(pdf_docs): # Extracting text from pdf
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) #It creates pdf objects that has pages:

        for page in pdf_reader.pages: # Looping through pages
            text += page.extract_text() #Extracting text from each page and appending to single string text

    return text   

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", #Seperate chunks by next line
        chunk_size = 1000, # character size of chunks
        chunk_overlap = 200, # Start from 200 characters before when starting next chunk to avoid info loss
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vectorstore(text_chunks): # Creating vector store using openAI Embeddings and storing locally in FAISS:
    #embeddings = OpenAIEmbeddings() # This method is paid and OpenAI Charges you
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts= text_chunks, embedding= embeddings )

    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI() # Llm is openai
    #llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512}, task="text2text-generation")
    llm = ChatOllama(
        model="llama3:latest",      # The model I pulled
        temperature=0.0
    )

    # Memory: that is our chat remembers the questions:
    memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_userinput(user_question):
    
    if st.session_state.conversation is None:
        st.error("⚠ Please upload documents and click 'Process' first.")
        return
    
    if not user_question.strip():
        return  # ignore empty messages
    
    response = st.session_state.conversation({'question': user_question}) # This conversation contain all config from our vectorstore and memory and will remember prev context
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 ==0: 
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
def summarize_pdf(raw_text):
    # Use the same Llama 3 model
    llm = ChatOllama(model="llama3:latest", temperature=0.0)
    
    prompt = f"Summarize the following text in a few sentences:\n\n{raw_text}"
    summary = llm.call_as_llm(prompt)  # returns the summary
    return summary

def main():

    load_dotenv() #Specific to langachain
    #Defining the GUI Layout:
    st.set_page_config(page_title = "SmartPDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # If application runs itself it will check if conversation is in session state or not:    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # **Initialize pdf_summary here!**
    if "pdf_summary" not in st.session_state:
        st.session_state.pdf_summary = ""


    st.header("SmartPDFs :books:")
    user_question = st.text_input("Ask a question about the documents:")

    # Only answer questions IF the chain exists
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs Here and 'Process'", accept_multiple_files=True)

        if "processed" not in st.session_state:
            st.session_state.processed = False

        if st.button("Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
                return

            with st.spinner("Processing Documents..."): # Everything inside the spinner is processed when we see loading logo
                # Get the pdf raw text - extracting:
                raw_text = get_pdf_text(pdf_docs) #Takes pdfs as input and returns single string of all contents
                #st.write(raw_text)

                short_text = raw_text[:3000]  # optional: limit text for summary

                # Getting the text chunks:
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # Create Vector store: ie creating embeddings from text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Generate PDF summary
                st.session_state.pdf_summary = summarize_pdf(raw_text)

                # Creating chain for Conversation: (keeping history of chats)
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # Generate Summary:
                st.session_state.pdf_summary = summarize_pdf(short_text)

                # Mark as processed
                st.session_state.processed = True

                if st.session_state.processed:
                    st.success("PDFs have been processed. You can now ask questions below.")

        if st.session_state.pdf_summary:
                    st.subheader("📄 PDF Summary")
                    with st.expander("Show Summary"):
                        st.info(st.session_state.pdf_summary)       
            
if __name__ == '__main__':
    main()