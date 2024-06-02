from pathlib import Path
from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from helpers.import_envs import openai_api_key, index_file, index_name
import pypandoc

def load_rtf_document(file_path):
    pypandoc.download_pandoc()
    # Load RTF file using LangChain's UnstructuredRTFLoader
    loader = UnstructuredRTFLoader(file_path)
    document = loader.load()
    return document


def load_rtf_document_and_chunk(file_path):
    pypandoc.download_pandoc()
    loader = UnstructuredRTFLoader(file_path)
    document = loader.load_and_split()  # uses RecursiveCharacterTextSplitter by default
    return document

def embed_chunks(chunked_docs):
    # create our embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=openai_api_key
    )  

    # create a local file store to for our cached embeddings
    store = LocalFileStore(
        "./cache/"
    )  
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, store, namespace=embedding_model.model
    )

    # Create vector store using Facebook AI Similarity Search (FAISS)
    vector_store = FAISS.from_documents(
        documents=chunked_docs, embedding=embedder
    )  # TODO: How do we create our vector store using FAISS?
    print(vector_store.index.ntotal)


    # save our vector store locally
    vector_store.save_local(folder_path=index_name)
    return vector_store

def create_or_load_vectore_store(transcript_file_name):
    chunked_docs = load_rtf_document_and_chunk(file_path=transcript_file_name)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=openai_api_key
    )  

    index_file_path = Path(index_file)
    if index_file_path.exists():
        print("Embeddings already done, use the saved index")
        # Combine the retrieved data with the output of the LLM
        vector_store = FAISS.load_local(
            index_name, embedding_model, allow_dangerous_deserialization=True
        )
    else:
        vector_store = embed_chunks(chunked_docs=chunked_docs)

    return vector_store
