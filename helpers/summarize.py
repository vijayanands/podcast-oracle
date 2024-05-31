from langchain.chains.summarize import load_summarize_chain
from helpers.prompts import BULLET_POINT_PROMPT

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
    output_summary = chain.invoke(docs)
    print(output_summary['output_text'])



def summarize_with_map_reduce(docs, llm):
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

    # prompt used by the chain for summarizing each part
    # print("prompt used by the chain for summarizing each part:")
    # print(chain.llm_chain.prompt.template)

    # prompt used by the chain for combining the parts
    # print("prompt used by the chain for combining the parts:")
    # print(chain.combine_document_chain.llm_chain.prompt.template)

    run_chain(chain=chain, docs=docs)


def summarize_with_map_reduce_and_bullet_point_prompt(docs, llm):
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=BULLET_POINT_PROMPT,
        combine_prompt=BULLET_POINT_PROMPT,
    )

    run_chain(chain=chain, docs=docs)


"""
Stuffing is the simplest method, whereby you simply stuff all the related data into the prompt as context to pass to 
the language model. This is implemented in LangChain as the StuffDocumentsChain.
Pros: Only makes a single call to the LLM. When generating text, the LLM has access to all the data at once.
Cons: Most LLMs have a context length, and for large documents (or many documents) this will not work as it will 
        result in a prompt larger than the context length.

The main downside of this method is that it only works one smaller pieces of data. Once you are working with many 
pieces of data, this approach is no longer feasible. The next two approaches are designed to help deal with that.
"""


def summarize_with_stuff_chain(docs, llm):
    chain = load_summarize_chain(llm, chain_type="stuff")
    run_chain(chain=chain, docs=docs)

    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_POINT_PROMPT)
    # run_chain(chain=chain, docs=docs)
