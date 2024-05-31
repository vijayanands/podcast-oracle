from helpers.utils import create_or_load_vectore_store
from helpers.import_envs import openai_api_key
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# Specify the path to the file you want to check
vector_store = create_or_load_vectore_store()

# create a prompt template to send to our LLM that will incorporate the documents from our retriever with the
# question we ask the chat model
prompt_template = ChatPromptTemplate.from_template(
    "Answer the {question} based on the following {context}."
)

# create a retriever for our documents
retriever = vector_store.as_retriever()

# create a chat model / LLM
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key
)

# create a parser to parse the output of our LLM
parser = StrOutputParser()

# 💻 Create the sequence (recipe)
runnable_chain = (
    # TODO: How do we chain the output of our retriever, prompt, model and model output parser so that we can get a good answer to our query?
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)

question = "What is the opinion of the speaker on open source?"
answer = runnable_chain.invoke(question)
print(answer)
