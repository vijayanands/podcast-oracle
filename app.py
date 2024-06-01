import gradio as gr
from helpers.utils import load_rtf_document_and_chunk
from helpers.summarize import summarize_with_map_reduce
from helpers.model_utils import get_model
from aspect_and_sentiment_extraction import extract_aspects_and_sentiment
from answer_bot import answer_question

def summarize(transcript_file_name):
    chunked_docs = load_rtf_document_and_chunk(transcript_file_name)

    llm = get_model("OPENAI")
    return summarize_with_map_reduce(chunked_docs, llm)

def extract_aspects(transcript_file_name):
    # Implement your aspect extraction and sentiment analysis logic here
    return extract_aspects_and_sentiment(transcript_file_name)

def get_answer_for(question):
    # Implement your QA logic here
    return answer_question(question)

def gradio_interface(uploaded_file, user_question):
    # Process the file
    transcript_file_name = uploaded_file.name
    print(f"Transcript File Name :{transcript_file_name}")
    
    # Summarize the content
    summary = summarize(transcript_file_name=transcript_file_name)

    # Aspect-Based Sentiment Analysis
    sentiment = extract_aspects(transcript_file_name=transcript_file_name)

    # Answer the user's question using the question-answering model
    if user_question.strip():  # Ensure there is a question provided
        answer_text = get_answer_for(question=user_question)
    else:
        answer_text = "No question asked."

    return summary, sentiment, answer_text

# Setup Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Transcript File"),
        gr.Textbox(label="Enter your question here", placeholder="Type your question...")
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Aspect-Based Sentiments"),
        gr.Textbox(label="Question Answering")
    ],
    title="Podcast Oracle",
    description="Upload your podcast transcript file to get a summary, aspect-based sentiments, and answers to predefined questions."
)

# Launch the app; this also hosts it on Hugging Face when run in the right environment
iface.launch()
