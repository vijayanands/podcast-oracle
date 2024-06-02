from helpers.import_envs import openai_api_key
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.schema import StrOutputParser
from helpers.model_utils import set_sentiment_analysis_llm, GPT3
import re

# Define the function to clean and extract text from RTF content
def extract_text_from_rtf(rtf_str):
    # Remove RTF tags and control words
    plain_text = re.sub(r'{\\[^{}]+}', '', rtf_str)
    plain_text = re.sub(r'\\[a-z]+\s?', '', plain_text)
    plain_text = plain_text.replace('\n', ' ').replace('\r', '')
    return plain_text

def extract_aspects_and_sentiment(transcript_file_name, llm_choice = None):
    sentiment_analysis_llm = set_sentiment_analysis_llm(llm_choice)
    # Read the RTF file content
    with open(transcript_file_name, 'r') as file:
        rtf_content = file.read()

    # Extract plain text from the RTF content
    document_text = extract_text_from_rtf(rtf_content)

    prompt = """

    Extract aspects from the given text {document}. Once the aspects are extracted print them as a bulleted list and then based on the nature of 
    aspects give a sentiment analysis on the aspects

    """

    prompt_template = PromptTemplate(template=prompt, input_variables=["document"])

    # create a chat model / LLM
    chat_model = ChatOpenAI(
        model=sentiment_analysis_llm.model_name, temperature=0, api_key=openai_api_key
    )

    # create a parser to parse the output of our LLM
    parser = StrOutputParser()

    # 💻 Create the sequence (recipe)
    runnable_chain = (
        {"document": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    answer = runnable_chain.invoke(document_text)
    print(answer)
    return answer
