# import os
# from PyPDF2 import PdfReader #later update to pypdf
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# import faiss
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv
# from langchain.chains import (
#     create_history_aware_retriever,
#     create_retrieval_chain,
# )

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain import hub

# # from langchain_core.vectorstores import InMemoryVectorStore

# # #memory
# # from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import InMemorySaver #long term memory
# from langgraph.store.memory import InMemoryStore #long term memory
# from langmem import create_manage_memory_tool, create_search_memory_tool # long term semantic memory
# # Define a schema for what you want to remember (customize as needed)
# from pydantic import BaseModel


# # vector_store = InMemoryVectorStore(embeddings=GoogleGenerativeAIEmbeddings())  


# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# from langmem import create_memory_store_manager

# class UserFact(BaseModel):
#     topic: str
#     value: str

# # Create a memory store manager
# memory_store = InMemoryStore(index={"dims": 1536, "embed": "gemini-embedding-exp-03-07"})
# memory_manager = create_memory_store_manager(
#     "gemini-1.5-flash",
#     schemas=[UserFact],
#     instructions="Extract and remember important facts from the conversation.",
#     enable_inserts=True,
#     enable_updates=True,
#     enable_deletes=False,
#     store=memory_store,
# )


# # extract text data
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# # convert text to chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=20,
#         length_function=len,
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# # convert chunks to embeddings
# def get_embeddings(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
#     return embeddings


# # store chunks in vector db faiss and save index
# def get_vector_store(chunks):
#     vector_store = FAISS.from_texts(chunks, embeddings = embeddings)
#     vector_store.save_local("faiss_index")
#     # Later...
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     return vector_store


# # create conversation chain
# # user query-> query embedding -> semantic search -> rank results -> generate response
# def get_conversation_chain_with_memory(vector_store, memory_manager):
#     llm = ChatGoogleGenerativeAI(
#         temperature=0,
#         model="gemini-1.5-flash",
#         max_output_tokens=512,
#         top_p=0.95,
#         top_k=40,
#     )
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#     qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
#     document_chain = create_stuff_documents_chain(llm, qa_prompt)
#     chain = create_retrieval_chain(retriever, document_chain)
#     return chain

# # Usage pattern:
# def chat_with_memory(chain, memory_manager, user_query, conversation_history):
#     # 1. Retrieve relevant memories for the query
#     memories = memory_manager.search(user_query)
#     memory_context = " ".join([m.value for m in memories])
#     # 2. Combine memory context with conversation history or prompt
#     full_input = f"{memory_context}\n{conversation_history}\n{user_query}"
#     # 3. Run the chain
#     response = chain.invoke({"input": full_input})
#     # 4. Extract new memories from the interaction and store
#     memory_manager.extract_memories(user_query + " " + response["answer"])
#     return response["answer"]

# def get_conversation_chain(vector_store):
#     llm = GooglePalm()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vector_store.as_retriever(), memory = memory)
#     return conversation_chain

import os
from pypdf import PdfReader  # remove pypdf with llama parse; causes hallucinations; not a good parser; especially for financial data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # checked GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langmem import create_memory_store_manager
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Define schema for memory storage 
class UserFact(BaseModel):
    topic: str
    value: str

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


# Create vector memory store for chatbot
def setup_memory():
    # Stores and searches memory using semantic similarity
    memory_store = InMemoryStore(index={
    "dims": 1536,
    "embed": embeddings.embed_documents  # This is the correct embedding function
})
# This creates a memory storage system in RAM (not on disk).
    
    # Extracts, inserts, and retrieves memory using the store
    memory_manager = create_memory_store_manager(
        "gemini-1.5-flash", # llm used to understand and extract user facts
        schemas=[UserFact], # user information schema
        instructions="Extract and remember important facts from the conversation.",
        enable_inserts=True,
        enable_updates=True,
        enable_deletes=False, # No deletes for now
        store=memory_store, # memory store
        #memory_saver=InMemorySaver(), # long term memory saver
    )
    return memory_manager

#Now for handling the documents

# Extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from text chunks
def get_vector_store(chunks):
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load existing vector store if available
def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create RAG chain with configured LLM
def create_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-1.5-flash",
        max_output_tokens=512,
        top_p=0.95,
        top_k=40,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain

# Process query with memory integration
def chat_with_memory(chain, memory_manager, user_query, conversation_history=""):
    # Retrieve relevant memories
    memories = memory_manager.search(user_query)
    memory_context = " ".join([m.value for m in memories]) if memories else ""
    
    # Combine memory with conversation context
    context = ""
    if memory_context:
        context += f"Relevant context from memory: {memory_context}\n\n"
    if conversation_history:
        context += f"Previous conversation: {conversation_history}\n\n"
    
    full_input = f"{context}User query: {user_query}"
    
    # Generate response using RAG
    response = chain.invoke({"input": full_input})
    
    # Store new memories from this interaction
    memory_manager.extract_memories(f"User: {user_query}\nAssistant: {response['answer']}")
    
    return response["answer"]

# Main workflow function
def process_documents_and_setup(pdf_files=None):
    """Set up the complete RAG system with memory"""
    memory_manager = setup_memory()
    
    try:
        # Try to load existing vector store
        print("Before loading vector store")
        vector_store = load_vector_store()
        print("After loading vector store")
        print("Loaded existing vector store.")
    except (FileNotFoundError, ValueError):
        # If not found, process documents and create new vector store
        if not pdf_files:
            raise ValueError("No existing vector store found and no PDF files provided.")
        
        print("Creating new vector store from documents...")
        text = get_pdf_text(pdf_files)
        chunks = get_text_chunks(text)
        vector_store = get_vector_store(chunks)
    
    # Create the RAG chain
    chain = create_rag_chain(vector_store)
    
    return chain, memory_manager

# Example usage
if __name__ == "__main__":
    # Example with new documents
    # chain, memory_manager = process_documents_and_setup(["document1.pdf", "document2.pdf"])
    
    # Example with existing vector store
    # chain, memory_manager = process_documents_and_setup()
    
    # Example chat interaction
    # response = chat_with_memory(chain, memory_manager, "What is the main topic in the documents?")
    # print(response)
    pass
