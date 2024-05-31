from langchain_community.document_loaders import UnstructuredRTFLoader
import pypandoc

def load_rtf_document(file_path):
    pypandoc.download_pandoc()
    # Load RTF file using LangChain's UnstructuredRTFLoader
    loader = UnstructuredRTFLoader(file_path)
    document = loader.load()
    return document


def load_rtf_document_and_chunk(file_path):
    pypandoc.download_pandoc()
    # Load RTF file using LangChain's UnstructuredRTFLoader
    loader = UnstructuredRTFLoader(file_path)
    document = loader.load_and_split()  # uses RecursiveCharacterTextSplitter by default
    return document
