import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import requests
import uuid

WAV2VEC = "wav2vec"
AUTOMODELFORSPEECH = "automodelforspeech"

class Audio_to_Text:
    def __init__(self):
        self.model_id = "openai/whisper-large-v3"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("Model loaded successfully.")
        
    def download_mp3(self, url, save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("MP3 file downloaded and saved successfully.")
    
    def convert_audio_to_text(self, audio_file, transcription_method):
        if transcription_method == WAV2VEC:
            return self.transcribe_audio_to_text_using_wav2vec(audio_file)
        else:
            transformers.logging.set_verbosity_info()
            result = self.pipe(audio_file, generate_kwargs={"language": "english"})
            print("Converted audio to text successfully.")
            # save the result to a text file
            return self.save_transcribed_text_to_file(result)
    
    def convert_audio_to_text_from_url(self, url, transcription_method):
        #get uuid for the audio file
        uuid_audio = str(uuid.uuid4())
        save_path = f"audio-{uuid_audio}.mp3"

        self.download_mp3(url, save_path)
        path_text_file_of_audio = self.convert_audio_to_text(save_path)
       
        return path_text_file_of_audio
    
    def save_transcribed_text_to_file(self, text):
        uuid_text = str(uuid.uuid4())
        save_file_name = f"transcript-{uuid_text}.txt"
        with open(save_file_name, "w") as file:
            file.write(text)
            print("Transcript saved successfully.")
        return save_file_name

    def transcribe_audio_to_text_using_wav2vec(self, mp3):
        asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
        text = asr(mp3)["text"]
        return self.save_transcribed_text_to_file(text)

    
def transcribe_podcast_from_mp3(mp3_file, transcription_method):
    audio_to_text = Audio_to_Text()
    return audio_to_text.convert_audio_to_text(mp3_file, transcription_method);

def transcribe_podcast(file_url, transcription_method):
    # Example usage:
    # url = "https://chrt.fm/track/138C95/prfx.byspotify.com/e/play.podtrac.com/npr-510310/traffic.megaphone.fm/NPR7010771664.mp3"
    
    audio_to_text = Audio_to_Text()    
    # Convert the audio file to text
    path_text_file_of_audio = audio_to_text.convert_audio_to_text_from_url(file_url, transcription_method)
    
    # Print the result
    print(path_text_file_of_audio)
    return path_text_file_of_audio