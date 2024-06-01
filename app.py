import gradio as gr
from helpers.utils import load_rtf_document_and_chunk
from helpers.summarize import summarize_with_map_reduce
from helpers.model_utils import get_model
from aspect_and_sentiment_extraction import extract_aspects_and_sentiment
from answer_bot import answer_question

transcript_file_name = None

def summarize(transcript_file_name):
    chunked_docs = load_rtf_document_and_chunk(transcript_file_name)

    llm = get_model("OPENAI")
    return summarize_with_map_reduce(chunked_docs, llm)

def extract_aspects(transcript_file_name):
    # Implement your aspect extraction and sentiment analysis logic here
    return extract_aspects_and_sentiment(transcript_file_name)


def get_answer_for(user_question):
    if transcript_file_name is None:
        return "No Transcript Uploaded, Upload RTF File First", ""
    
    # Answer the user's question using the question-answering model
    if user_question.strip():  # Ensure there is a question provided
        answer_text = answer_question(question=user_question)
    else:
        answer_text = "No question asked."

    return answer_text.lstrip()

def process_transcript(uploaded_file):
    if transcript_file_name is None:
        return "No Transcript Uploaded, Upload RTF File First", ""
    
    # Summarize the content
    summary = summarize(transcript_file_name=transcript_file_name).lstrip()

    # Aspect-Based Sentiment Analysis
    sentiment = extract_aspects(transcript_file_name=transcript_file_name).lstrip()

    return summary, sentiment

def setup_rtf_file_handle(uploaded_file):
    transcript_file_name = uploaded_file.name
    print(f"Transcript File Name :{transcript_file_name}")

with gr.Blocks() as demo:
    with gr.Group("Upload RTF File"):
        rtf_file = gr.File(label="Podcast Transcript RTF file")
        submit_button = gr.Button("Upload File")
        submit_button.click(setup_rtf_file_handle)
    with gr.Group("Aspects and Sentiment of Podcast"):
        summary = gr.Textbox(label="Summary of Podcast")
        sentiment = gr.Textbox(label="Aspect Based Sentiments")
        submit_button = gr.Button("Generate Aspects and Summary")
        submit_button.click(process_transcript, inputs=rtf_file, outputs=[summary, sentiment])    

    with gr.Group("Question/Answer"):
        gr.Markdown("Question/Answer")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        answer_button = gr.Button("Answer Question")
        answer_button.click(get_answer_for, inputs= question, outputs=answer)

demo.launch()
