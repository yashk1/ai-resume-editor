# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core import Document  # Add this line


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def compare_resume_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity * 100  # Convert to percentage

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

with st.sidebar:
    st.header("Upload Resume and Job Description")
    
    uploaded_resume = st.file_uploader("Upload your resume (PDF)", type="pdf")
    job_description = st.text_area("Paste the job description here")

    if uploaded_resume and job_description:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_resume.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_resume.getvalue())
                
                # Load resume
                loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"])
                resume_docs = loader.load_data()
                resume_text = " ".join([doc.text for doc in resume_docs])
                
                # Create a document for the job description
                job_doc = Document(text=job_description, metadata={"source": "Job Description"})
                
                # Combine resume and job description documents
                all_docs = resume_docs + [job_doc]
                
                similarity_score = compare_resume_job(resume_text, job_description)
                
                st.success(f"Resume Match Score: {similarity_score:.2f}%")
                display_pdf(uploaded_resume)

                # Initialize query_engine with both resume and job description
                llm = Ollama(model="llama3.1", request_timeout=120.0)
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                Settings.embed_model = embed_model
                index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
                Settings.llm = llm
                st.session_state.query_engine = index.as_query_engine(streaming=True)

                # Customize prompt template
                qa_prompt_tmpl_str = (
                    "Context information is below. This includes details from the resume and the job description.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above, which includes both the resume and job description, "
                    "answer the query about the resume, job description, or their relationship. "
                    "If you don't know or the information is not provided, say 'I don't have enough information to answer that.'\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                st.session_state.query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"AI Resume editor - Llama-3.1 locally")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Modify the chat input and response section
if prompt := st.chat_input("Ask about the resume or job description"):
    if st.session_state.query_engine is None:
        st.error("Please upload a resume and provide a job description first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            streaming_response = st.session_state.query_engine.query(prompt)
            
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})