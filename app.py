from utils import load_rtf_document_and_chunk, load_rtf_document
from summarize import summarize_with_map_reduce

unchunked_documents = load_rtf_document(file_path=file_path)
# print("Printing Length of unchunked documents loaded")
# for doc in unchunked_documents:
#     print(len(doc.page_content))
#     print(doc.page_content)

chunked_documents = load_rtf_document_and_chunk(file_path=file_path)
# print("Printing Length of chunked documents loaded")
# for doc in chunked_documents:
#     print(len(doc.page_content))
#     print(doc.page_content)

summarize_with_map_reduce(chunked_documents)
# summarize_with_stuff_chain(unchunked_documents)
# summarize_with_map_reduce_and_bullet_point_prompt(chunked_documents)