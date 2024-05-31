from helpers.utils import load_rtf_document_and_chunk, load_rtf_document
from helpers.summarize import summarize_with_map_reduce
from helpers.model_utils import get_model
from helpers.import_envs import rtf_file

llm = get_model()

unchunked_documents = load_rtf_document(file_path=rtf_file)
# print("Printing Length of unchunked documents loaded")
# for doc in unchunked_documents:
#     print(len(doc.page_content))
#     print(doc.page_content)

chunked_documents = load_rtf_document_and_chunk(file_path=rtf_file)
# print("Printing Length of chunked documents loaded")
# for doc in chunked_documents:
#     print(len(doc.page_content))
#     print(doc.page_content)

summarize_with_map_reduce(chunked_documents, llm)
# summarize_with_stuff_chain(unchunked_documents, llm)
# summarize_with_map_reduce_and_bullet_point_prompt(chunked_documents, llm)