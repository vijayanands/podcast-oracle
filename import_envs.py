import os
from dotenv import load_dotenv
# llm_model="OPENAI"
llm_model="LLAMA3"
# llm_model_NAME="CLAUDE"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HF_HUB_TOKEN")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
