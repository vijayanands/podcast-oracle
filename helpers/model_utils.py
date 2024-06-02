from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from helpers.import_envs import openai_api_key, anthropic_api_key

GPT3 = "gpt-3.5"
GPT3_INSTRUCT = "gpt-3.5-instruct"
GPT4 = "gpt-4o"
LLAMA3 = "Llama3"
ANTHROPIC = "Claude2"

def _set_llm_based_on_choice(choice):
    if choice == GPT3_INSTRUCT:
        model_name = "gpt-3.5-turbo-instruct"
        llm = OpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    elif choice == GPT3:
        model_name = "gpt-3.5-turbo"
        llm = OpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    elif choice == GPT4:
        model_name = "gpt-4o"
        llm = OpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    elif choice == ANTHROPIC:
        model_name = "clause-2.1"
        llm = ChatAnthropic(model_name=model_name, anthropic_api_key=anthropic_api_key)
    elif choice == LLAMA3:
        model_name = "llama3"
        llm = Ollama(model=model_name)
    else:
        model_name = "gpt-3.5-turbo"
        llm = OpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    return llm

def set_summarization_llm(choice = None):
    return _set_llm_based_on_choice(choice)

def set_sentiment_analysis_llm(choice = None):
    return _set_llm_based_on_choice(choice)

def set_question_answer_llm(choice = None):
    return _set_llm_based_on_choice(choice)

