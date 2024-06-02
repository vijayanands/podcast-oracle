import os
from dotenv import load_dotenv

index_name = "podcast_oracle_index"
index_file = f"./{index_name}/index.faiss"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HF_HUB_TOKEN")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
