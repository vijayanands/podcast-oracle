import gradio as gr
from helpers.model_utils import GPT3, GPT4, LLAMA3, ANTHROPIC2, set_question_answer_llm, set_sentiment_analysis_llm, set_summarization_llm
from tools.summarize import MAPREDUCE, STUFF, summarize_podcast
from tools.answer_bot import answer_question
from tools.aspect_and_sentiment_extraction import extract_aspects_and_sentiment

def get_answer_for(user_question, transcript_file_name, question_answer_llm_choice):
    if transcript_file_name is None:
        answer_text = "No Transcript Uploaded, Upload RTF File First", ""
    elif not question_answer_llm_choice:
        answer_text = "No LLM Selected, select one"
    elif not user_question:
        answer_text = "Question Not Given"
    else:    
        # Answer the user's question using the question-answering model
        answer_text = answer_question(question=user_question, transcript_file_name=transcript_file_name, llm_choice=question_answer_llm_choice)

    return answer_text.lstrip(), transcript_file_name, question_answer_llm_choice

def summarize(uploaded_file, transcript_file_name, summarization_method, summarization_llm_choice):
    if transcript_file_name is None:
        summary = "No Transcript Uploaded, Upload RTF File First", ""
    elif not summarization_llm_choice:
        summary = "No LLM Selected, select one"
    elif not summarization_method:
        summary = "No Summarization Method Selected , select one"
    else:        
        # Summarize the content
        summary = summarize_podcast(transcript_file_name=transcript_file_name, summarization_method=summarization_method, llm_choice=summarization_llm_choice).lstrip()

    return summary, transcript_file_name, summarization_method, summarization_llm_choice

def generate_aspects_and_sentiments(uploaded_file, transcript_file_name, sentiment_analysis_llm_choice):
    if transcript_file_name is None:
        sentiment = "No Transcript Uploaded, Upload RTF File First", ""
    elif not sentiment_analysis_llm_choice:
        sentiment = "No LLM Selected, select one"
    else:
        # Aspect-Based Sentiment Analysis
        sentiment = extract_aspects_and_sentiment(transcript_file_name=transcript_file_name, llm_choice=sentiment_analysis_llm_choice).lstrip()

    return sentiment, transcript_file_name, sentiment_analysis_llm_choice

def setup_rtf_file_handle(uploaded_file, transcript_file_name):
    if not uploaded_file:
        status = "No File Detected, Failure"
    else:
        transcript_file_name = uploaded_file.name
        status = "Upload Success"
    return status, transcript_file_name

def setup_summarization_llm(choice, summarization_llm_choice):
    set_summarization_llm(choice)
    summarization_llm_choice = choice
    return choice, summarization_llm_choice

def setup_sentiment_analysis_llm(choice, sentiment_analysis_llm_choice):
    set_sentiment_analysis_llm(choice)
    sentiment_analysis_llm_choice = choice
    return choice, sentiment_analysis_llm_choice

def setup_question_answer_llm(choice, question_answer_llm_choice):
    set_question_answer_llm(choice)
    question_answer_llm_choice = choice
    return choice, question_answer_llm_choice

def setup_summarization_method(choice, summarization_method):
    summarization_method = choice
    return choice, summarization_method

def transcribe_audio_file(audio_file_link):
    return
    
summarization_llm_choices = [GPT3, GPT4, ANTHROPIC2]
question_answer_llm_choices = [GPT3, GPT4, ANTHROPIC2]
sentiment_analysis_llm_choices = [GPT3, GPT4, ANTHROPIC2]
summarize_method_choices = [MAPREDUCE, STUFF]

with gr.Blocks() as demo:
    transcript_file_name = gr.State()
    summarization_method = gr.State()
    question_answer_llm_choice = gr.State()
    sentiment_analysis_llm_choice = gr.State()
    summarization_llm_choice = gr.State()
    with gr.Group("Upload RTF File"):
        rtf_file = gr.File(label="Podcast Transcript RTF file")
        submit_button = gr.Button("Upload File")
        submit_status = gr.Textbox(label="", value="Pending Upload")
        submit_button.click(setup_rtf_file_handle, inputs=[rtf_file, transcript_file_name], outputs=[submit_status, transcript_file_name])
    with gr.Group("LLM Selection"):
        with gr.Row():
            choice = gr.Radio(label="Summarization LLM", choices=summarization_llm_choices)
            output = gr.Textbox(label="")
            choice.change(setup_summarization_llm, inputs=[choice,summarization_llm_choice], outputs=[output,summarization_llm_choice])
        with gr.Row():
            choice = gr.Radio(label="Sentiment Analysis LLM", choices=sentiment_analysis_llm_choices)
            output = gr.Textbox(label="")
            choice.change(setup_sentiment_analysis_llm, inputs=[choice,sentiment_analysis_llm_choice], outputs=[output,sentiment_analysis_llm_choice])
        with gr.Row():
            choice = gr.Radio(label="Question/Answer LLM", choices=question_answer_llm_choices)
            output = gr.Textbox(label="")
            choice.change(setup_question_answer_llm, inputs=[choice,question_answer_llm_choice], outputs=[output,question_answer_llm_choice])
    with gr.Group("Summarization Method"):
        choice = gr.Radio(label="Summarization Method", choices=summarize_method_choices)
        output = gr.Textbox(label="")
        choice.change(setup_summarization_method, inputs=[choice, summarization_method], outputs=[output, summarization_method])
    with gr.Group("Summarize Podcast"):     
        summary = gr.Textbox(label="Summary of Podcast")
        submit_button = gr.Button("Generate Summary")
        submit_button.click(summarize, inputs=[rtf_file, transcript_file_name, summarization_method, summarization_llm_choice], outputs=[summary, transcript_file_name, summarization_method, summarization_llm_choice])    
    with gr.Group("Aspects and Sentiment of Podcast"):     
        sentiment = gr.Textbox(label="Aspect Based Sentiments")
        submit_button = gr.Button("Generate Aspects and Summary")
        submit_button.click(generate_aspects_and_sentiments, inputs=[rtf_file, transcript_file_name, sentiment_analysis_llm_choice], outputs=[sentiment, transcript_file_name, sentiment_analysis_llm_choice])    
    with gr.Group("Question/Answer"):
        gr.Markdown("Question/Answer")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        answer_button = gr.Button("Answer Question")
        answer_button.click(get_answer_for, inputs=[question, transcript_file_name, question_answer_llm_choice], outputs=[answer, transcript_file_name, question_answer_llm_choice])

demo.launch()
