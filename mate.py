import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Use Hugging Face for embeddings
from langchain_groq import ChatGroq  # ✅ Correct Import
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch API Key from .env (Fixing the case sensitivity issue)
api_key = os.getenv("GROQ_API_KEY")

# ✅ Function to read PDFs and return extracted text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ✅ Function to split text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ✅ Function to create vector store using FAISS
def get_vector_store(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ Use a HuggingFace model
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error during vector store creation: {e}")
        st.stop()

# ✅ Function to create conversational AI chain
def get_conversational_chain(user_question):
    try:
        prompt_template = """
        Answer the question in detail based on the provided context. 
        If the answer is not in the context, say "Answer not found in context."
        
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """

        model = ChatGroq(
            model_name="llama3-8b-8192",  # ✅ Correct Groq model
            api_key=api_key,
            temperature=0.3
        )

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

        return chain
    except Exception as e:
        st.error(f"Error during conversational chain creation: {e}")
        st.stop()

# ✅ Function to process user input and generate a response
def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        st.error(f"Error during user input processing: {e}")
        return None

# ✅ Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Groq PDF Chatbot", page_icon="🤖")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("EXPLORER:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! Ask your questions.")
            else:
                st.error("Please upload at least one PDF.")

    # Chat interface
    st.title("Chat with PDFs using Groq 🤖")

    # Load previous chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                if response:
                    full_response = response['output_text']
                    st.write(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No response generated. Please try again.")

if __name__ == "__main__":
    main()
