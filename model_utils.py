from import_envs import llm_model, openai_api_key, anthropic_api_key
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

default_model = OpenAI(temperature=0, api_key=openai_api_key)

def get_model():
    if llm_model == "OPENAI":
        llm = default_model
        print(f"Model Name: {llm.model_name}");
    elif llm_model == "CLAUDE":
        llm = ChatAnthropic(model_name="claude-2.1", anthropic_api_key=anthropic_api_key)
        print(f"Model Name: {llm.model}");
    elif llm_model == "LLAMA3":
        # Now you can use `llm` for generating responses, etc.
        llm = Ollama(model="llama3")
        print(f"Model Name: {llm.model}");
    else:
        llm = default_model
        print(f"Model Name: {llm.model_name}");
    return llm
