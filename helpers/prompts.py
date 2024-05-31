from langchain.prompts import PromptTemplate

prompt_template = """
Write a concise bullet point summary of the following:

{text}

CONSCISE SUMMARY IN BULLET POINTS:
"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

prompt_template = """
Write a concise summary of the following:

{text}

CONSCISE SUMMARY IN BULLET POINTS:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
