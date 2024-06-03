from langchain.chains.summarize import load_summarize_chain
from helpers.prompts import BULLET_POINT_PROMPT
from helpers.utils import load_rtf_document_and_chunk, load_rtf_document
from helpers.model_utils import set_summarization_llm

MAPREDUCE="map-reduce"
STUFF="stuff"
summarization_method = MAPREDUCE

"""
This method involves an initial prompt on each chunk of data * ( for summarization tasks, this could be a summary 
of that chunk; for question-answering tasks, it could be an answer based solely on that chunk). Then a different 
prompt is run to combine all the initial outputs. This is implemented in the LangChain as the 
MapReduceDocumentsChain.
Pros: Can scale to larger documents (and more documents) than StuffDocumentsChain. The calls to the LLM on 
        individual documents are independent and can therefore be parallelized.
Cons: Requires many more calls to the LLM than StuffDocumentsChain. Loses some information during the final 
        combining call.
"""

def run_chain(chain, docs):
    output = chain.invoke(docs)
    summary = output['output_text']
    print(summary)
    return summary
    # prompt used by the chain for summarizing each part
    # print("prompt used by the chain for summarizing each part:")
    # print(chain.llm_chain.prompt.template)

    # prompt used by the chain for combining the parts
    # print("prompt used by the chain for combining the parts:")
    # print(chain.combine_document_chain.llm_chain.promdocs


def _summarize_with_map_reduce(transcript_file_name, llm):
    chunked_docs = load_rtf_document_and_chunk(transcript_file_name)
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

    # prompt used by the chain for summarizing each part
    # print("prompt used by the chain for summarizing each part:")
    # print(chain.llm_chain.prompt.template)

    # prompt used by the chain for combining the parts
    # print("prompt used by the chain for combining the parts:")
    # print(chain.combine_document_chain.llm_chain.prompt.template)

    return run_chain(chain=chain, docs=chunked_docs)

def _summarize_with_map_reduce_and_bullet_point_prompt(transcript_file_name, llm):
    chunked_docs = load_rtf_document_and_chunk(transcript_file_name)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=BULLET_POINT_PROMPT,
        combine_prompt=BULLET_POINT_PROMPT,
    )

    return run_chain(chain=chain, docs=chunked_docs)


"""
Stuffing is the simplest method, whereby you simply stuff all the related data into the prompt as context to pass to 
the language model. This is implemented in LangChain as the StuffDocumentsChain.
Pros: Only makes a single call to the LLM. When generating text, the LLM has access to all the data at once.
Cons: Most LLMs have a context length, and for large documents (or many documen# extract_aspects_and_sentiment(rtf_file)
s) this will not work as it will 
        result in a prompt larger than the context length.

The main downside of this method is that it only works one smaller pieces of data. Once you are working with many 
pieces of data, this approach is no longer feasible. The next two approaches are designed to help deal with that.
"""


def _summarize_with_stuff_chain(transcript_file_name, llm):
    docs = load_rtf_document(transcript_file_name)
    chain = load_summarize_chain(llm=llm, chain_type="stuff")
    return run_chain(chain=chain, docs=docs)

    # chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT)
    # run_chain(chain=chain, docs=docs)


def summarize_podcast(transcript_file_name, summarization_method = None, llm_choice = None):
    llm = set_summarization_llm(llm_choice)
    if summarization_method == MAPREDUCE:
        return _summarize_with_map_reduce(transcript_file_name=transcript_file_name, llm=llm)
    elif summarization_method == STUFF:
        return _summarize_with_stuff_chain(transcript_file_name=transcript_file_name, llm=llm)
    else:
        return _summarize_with_map_reduce(transcript_file_name=transcript_file_name, llm=llm)