import os
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from helpers.import_envs import openai_api_key, anthropic_api_key, huggingface_token
from langchain_openai import ChatOpenAI 
from transformers.pipelines import pipeline
# from langchain_community.llms.openllm import OpenLLM

GPT3 = "gpt-3.5"
GPT4 = "gpt-4o"
LLAMA3 = "meta-llama/Meta-Llama-3-8B"
ANTHROPIC2 = "Claude-2.1"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"

def _set_llm_based_on_choice(choice):
    if choice == GPT3:
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    elif choice == GPT4:
        model_name = "gpt-4o"
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    elif choice == ANTHROPIC2:
        model_name = "claude-2.1"
        llm = ChatAnthropic(model_name=model_name, anthropic_api_key=anthropic_api_key)
    elif choice == LLAMA3:
        model_name = LLAMA3
        llm = pipeline("text-generation", model=model_name, token=huggingface_token)
    # elif choice == MISTRAL:
    #     runpod_endpoint = "https://api.runpod.ai/v2/q67259l60h6adh/openai/v1"
    #     runpod_api_key = os.getenv("RUNPOD_API_KEY")
    #     gen_kwargs = {
    #         "temperature": 0,
    #         "api_key": runpod_api_key
    #     }
    #     server_url = runpod_endpoint  # Replace with remote host if you are running on a remote server
    #     llm = OpenLLM(server_url=server_url, model_name=MISTRAL, llm_kwargs=gen_kwargs) 
    else:
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=openai_api_key)
    return llm

def set_summarization_llm(choice = None):
    return _set_llm_based_on_choice(choice)

def set_sentiment_analysis_llm(choice = None):
    return _set_llm_based_on_choice(choice)

def set_question_answer_llm(choice = None):
    return _set_llm_based_on_choice(choice)

