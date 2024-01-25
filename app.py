from openai_api_key import apikey
import os   # Import os to set API key
from langchain_openai import OpenAI
# UI/app interface
import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader  # Inport PDF document loaders...
from langchain_community.vectorstores.chroma import Chroma  # import chroma as vector store

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# Bring in streamlit for UI/app interface


# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = apikey

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings(embedding_ctx_length=300000)

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Investment Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.invoke(prompt)

    output_text = response.get("output", "")

    st.write(output_text)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 