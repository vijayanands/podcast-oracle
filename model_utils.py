from import_envs import llm_model, openai_api_key, anthropic_api_key
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

default_model = OpenAI(temperature=0, api_key=openai_api_key)

def get_model(model_override = None):
    if model_override is not None:
        model_str = model_override
    else:
        model_str = llm_model
    if model_str == "OPENAI":
        llm = default_model
        print(f"Model Name: {llm.model_name}");
    elif model_str == "CLAUDE":
        llm = ChatAnthropic(model_name="claude-2.1", anthropic_api_key=anthropic_api_key)
        print(f"Model Name: {llm.model}");
    elif model_str == "LLAMA3":
        # Now you can use `llm` for generating responses, etc.
        llm = Ollama(model="llama3")
        print(f"Model Name: {llm.model}");
    else:
        llm = default_model
        print(f"Model Name: {llm.model_name}");
    return llm
