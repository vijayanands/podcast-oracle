import gradio as gr
from helpers.model_utils import GPT3, GPT4, LLAMA3, ANTHROPIC2, MISTRAL, set_question_answer_llm, set_sentiment_analysis_llm, set_summarization_llm
from tools.summarize import MAPREDUCE, STUFF, summarize_podcast
from tools.answer_bot import answer_question
from tools.aspect_and_sentiment_extraction import extract_aspects_and_sentiment
from tools.transcribe import transcribe_podcast, transcribe_podcast_from_mp3, WAV2VEC, WHISPER

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

def summarize(transcript_file_name, summarization_method, summarization_llm_choice):
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

def generate_aspects_and_sentiments(transcript_file_name, sentiment_analysis_llm_choice):
    if transcript_file_name is None:
        sentiment = "No Transcript Uploaded, Upload RTF File First", ""
    elif not sentiment_analysis_llm_choice:
        sentiment = "No LLM Selected, select one"
    else:
        # Aspect-Based Sentiment Analysis
        sentiment = extract_aspects_and_sentiment(transcript_file_name=transcript_file_name, llm_choice=sentiment_analysis_llm_choice).lstrip()

    return sentiment, transcript_file_name, sentiment_analysis_llm_choice

def setup_transcript_file_handle(uploaded_file, transcript_file_name):
    if not uploaded_file:
        transcription_status = "No File Detected, Failure"
    else:
        transcript_file_name = uploaded_file.name
        transcription_status = "Upload Success"
    return transcription_status, transcript_file_name

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

def setup_transcription_method(choice, transcription_method):
    transcription_method = choice
    return choice, transcription_method

def get_transcribed_text_from_file(transcription_file):
    # Define a variable to hold the content
    file_content = ""

    # Open the file in read mode
    with open(transcription_file, 'r') as file:
        # Read the entire content of the file into the string variable
        file_content = file.read()

    return file_content

def transcribe_audio_file(uploaded_file, transcript_file_name, transcription_method):
    if not uploaded_file:
        status = "No File Detected, Failure"
    else:
        transcript_file_name = transcribe_podcast_from_mp3(uploaded_file.name, transcription_method)
        status = "Upload Success"
    return transcript_file_name, transcription_method, get_transcribed_text_from_file(transcript_file_name)

def download_and_transcribe_podcast(mp3_url, transcript_file, transcription_method):
    if not mp3_url:
        status = "No URL detected, Failure"
    else:
        transcript_file = transcribe_podcast(mp3_url, transcription_method)
        status = "Upload Success"
    return transcript_file, transcription_method, get_transcribed_text_from_file(transcript_file_name)
    
summarization_llm_choices = [GPT3, GPT4, ANTHROPIC2, MISTRAL]
question_answer_llm_choices = [GPT3, GPT4, ANTHROPIC2]
sentiment_analysis_llm_choices = [GPT3, GPT4, ANTHROPIC2]
summarize_method_choices = [MAPREDUCE, STUFF]
transcription_method_choices = [WAV2VEC, WHISPER]

with gr.Blocks() as demo:
    transcript_file = gr.State()
    summarization_method = gr.State()
    question_answer_llm_choice = gr.State()
    sentiment_analysis_llm_choice = gr.State()
    summarization_llm_choice = gr.State()
    transcription_method = gr.State(value=WHISPER)

    # with gr.Group("Trancsription Model Selection"):
    #     with gr.Row():
    #         choice = gr.Radio(label="Transcription Model", choices=transcription_method_choices, value=WAV2VEC)
    #         output = gr.Textbox(label="")
    #         choice.change(setup_transcription_method, inputs=[choice, transcription_method], outputs=[output, transcription_method])
    with gr.Group("Enter Podcast mp3 URL"):
        mp3_url = gr.Textbox(label="Podcast MP3 URL")
        submit_button = gr.Button("Transcribe")
        transcript = gr.Textbox(label="Transcript of Podcast")
        submit_button.click(download_and_transcribe_podcast, inputs=[mp3_url, transcript_file, transcription_method], outputs=[transcript_file, transcription_method, transcript])
    with gr.Group("Upload Podcast mp3 File"):
        mp3_file = gr.File(label="Podcast mp3 file")
        submit_button = gr.Button("Transcribe")
        transcript = gr.Textbox(label="Transcript of Podcast")
        submit_button.click(transcribe_audio_file, inputs=[mp3_file, transcript_file, transcription_method], outputs=[transcript_file, transcription_method, transcript])
    with gr.Group("Upload RTF File"):
        rtf_file = gr.File(label="Transcripted RTF file")
        submit_button = gr.Button("Upload RTF")
        status = gr.Textbox(label="", value="Pending Upload")
        submit_button.click(setup_transcript_file_handle, inputs=[rtf_file, transcript_file], outputs=[status, transcript_file])
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
        submit_button.click(summarize, inputs=[transcript_file, summarization_method, summarization_llm_choice], outputs=[summary, transcript_file, summarization_method, summarization_llm_choice])    
    with gr.Group("Aspects and Sentiment of Podcast"):     
        sentiment = gr.Textbox(label="Aspect Based Sentiments")
        submit_button = gr.Button("Generate Aspects and Summary")
        submit_button.click(generate_aspects_and_sentiments, inputs=[transcript_file, sentiment_analysis_llm_choice], outputs=[sentiment, transcript_file, sentiment_analysis_llm_choice])    
    with gr.Group("Question/Answer"):
        gr.Markdown("Question/Answer")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        answer_button = gr.Button("Answer Question")
        answer_button.click(get_answer_for, inputs=[question, transcript_file, question_answer_llm_choice], outputs=[answer, transcript_file, question_answer_llm_choice])

demo.launch()
