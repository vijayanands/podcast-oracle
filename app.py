import gradio as gr
from helpers.model_utils import GPT3, GPT4, LLAMA3, ANTHROPIC, set_question_answer_llm, set_sentiment_analysis_llm, set_summarization_llm
from tools.summarize import MAPREDUCE, STUFF, summarize_podcast
from tools.answer_bot import answer_question
from tools.aspect_and_sentiment_extraction import extract_aspects_and_sentiment

def get_answer_for(user_question, transcript_file_name, llm_choice):
    if transcript_file_name is None:
        return "No Transcript Uploaded, Upload RTF File First", ""
    if user_question is None:
        return "Question Not Given"
    
    # Answer the user's question using the question-answering model
    if user_question.strip():  # Ensure there is a question provided
        answer_text = answer_question(question=user_question, transcript_file_name=transcript_file_name, llm_choice=llm_choice)
    else:
        answer_text = "No question asked."

    return answer_text.lstrip(), transcript_file_name, llm_choice

def summarize(uploaded_file, transcript_file_name, summarization_method, llm_choice):
    if transcript_file_name is None:
        return "No Transcript Uploaded, Upload RTF File First", ""
    
    # Summarize the content
    summary = summarize_podcast(transcript_file_name=transcript_file_name, summarization_method=summarization_method, llm_choice=llm_choice).lstrip()

    return summary, transcript_file_name, summarization_method, llm_choice

def generate_aspects_and_sentiments(uploaded_file, transcript_file_name, llm_choice):
    if transcript_file_name is None:
        return "No Transcript Uploaded, Upload RTF File First", ""
    
    # Aspect-Based Sentiment Analysis
    sentiment = extract_aspects_and_sentiment(transcript_file_name=transcript_file_name, llm_choice=llm_choice).lstrip()

    return sentiment, transcript_file_name, llm_choice

def setup_rtf_file_handle(uploaded_file, transcript_file_name):
    if not uploaded_file:
        return None
    transcript_file_name = uploaded_file.name
    return transcript_file_name

def setup_summarization_llm(choice, llm_choice):
    set_summarization_llm(choice)
    llm_choice = choice
    return choice, llm_choice

def setup_sentiment_analysis_llm(choice, llm_choice):
    set_sentiment_analysis_llm(choice)
    llm_choice = choice
    return choice, llm_choice

def setup_question_answer_llm(choice, llm_choice):
    set_question_answer_llm(choice)
    llm_choice = choice
    return choice, llm_choice

def setup_summarization_method(choice, summarization_method):
    summarization_method = choice
    return choice, summarization_method

    
llm_choices = [GPT3, GPT4, LLAMA3, ANTHROPIC]
summarize_method_choices = [MAPREDUCE, STUFF]

with gr.Blocks() as demo:
    transcript_file_name = gr.State()
    summarization_method = gr.State()
    llm_choice = gr.State()
    with gr.Group("Upload RTF File"):
        rtf_file = gr.File(label="Podcast Transcript RTF file")
        submit_button = gr.Button("Upload File")
        submit_button.click(setup_rtf_file_handle, inputs=[rtf_file, transcript_file_name], outputs=transcript_file_name)
    with gr.Group("LLM Selection"):
        with gr.Row():
            choice = gr.Radio(label="Summarization LLM", choices=llm_choices, value=GPT3)
            output = gr.Textbox(label="", value=GPT3)
            choice.change(setup_summarization_llm, inputs=[choice,llm_choice], outputs=[output,llm_choice])
        with gr.Row():
            choice = gr.Radio(label="Sentiment Analysis LLM", choices=llm_choices, value=GPT3)
            output = gr.Textbox(label="", value=GPT3)
            choice.change(setup_summarization_llm, inputs=[choice,llm_choice], outputs=[output,llm_choice])
        with gr.Row():
            choice = gr.Radio(label="Question/Answer LLM", choices=llm_choices, value=GPT3)
            output = gr.Textbox(label="", value=GPT3)
            choice.change(setup_summarization_llm, inputs=[choice,llm_choice], outputs=[output,llm_choice])
    with gr.Group("Summarization Method"):
        choice = gr.Radio(label="Summarization Method", choices=summarize_method_choices, value=MAPREDUCE)
        output = gr.Textbox(label="", value=MAPREDUCE)
        choice.change(setup_summarization_method, inputs=[choice, summarization_method], outputs=[output, summarization_method])
    with gr.Group("Summarize Podcast"):     
        summary = gr.Textbox(label="Summary of Podcast")
        submit_button = gr.Button("Generate Summary")
        submit_button.click(summarize, inputs=[rtf_file, transcript_file_name, summarization_method, llm_choice], outputs=[summary, transcript_file_name, summarization_method, llm_choice])    
    with gr.Group("Aspects and Sentiment of Podcast"):     
        sentiment = gr.Textbox(label="Aspect Based Sentiments")
        submit_button = gr.Button("Generate Aspects and Summary")
        submit_button.click(generate_aspects_and_sentiments, inputs=[rtf_file, transcript_file_name, llm_choice], outputs=[sentiment, transcript_file_name, llm_choice])    
    with gr.Group("Question/Answer"):
        gr.Markdown("Question/Answer")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        answer_button = gr.Button("Answer Question")
        answer_button.click(get_answer_for, inputs=[question, transcript_file_name, llm_choice], outputs=[answer, transcript_file_name, llm_choice])

demo.launch()
