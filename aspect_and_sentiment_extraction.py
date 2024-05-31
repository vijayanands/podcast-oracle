from helpers.import_envs import openai_api_key
from helpers.import_envs import rtf_file
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.schema import StrOutputParser
import re

# Define the function to clean and extract text from RTF content
def extract_text_from_rtf(rtf_str):
    # Remove RTF tags and control words
    plain_text = re.sub(r'{\\[^{}]+}', '', rtf_str)
    plain_text = re.sub(r'\\[a-z]+\s?', '', plain_text)
    plain_text = plain_text.replace('\n', ' ').replace('\r', '')
    return plain_text


# Read the RTF file content
with open(rtf_file, 'r') as file:
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
    model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key
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
