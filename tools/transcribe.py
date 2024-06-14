from transformers import pipeline
import requests
import uuid
from pydub import AudioSegment
import os
import whisper

class Audio_to_Text:
    def __init__(self):
        print("Model loaded successfully.")
    
    def split_audio(self, file_path, segment_length_ms= 5*60*1000):
        # Load the audio file
        audio = AudioSegment.from_file(file_path, format="mp3")

         # Calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        segments = []
        for i in range(num_segments):
            start = i * segment_length_ms
            end = start + segment_length_ms
            segment = audio[start:end]
            segments.append(segment)

        # Handle the last segment if there is a remainder
        if len(audio) % segment_length_ms != 0:
            segments.append(audio[num_segments * segment_length_ms:])

        return segments
        
    def download_mp3(self, url, save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("MP3 file downloaded and saved successfully.")
        
    def convert_audio_to_text_by_splitting(self, audio_file):
        segments = self.split_audio(audio_file)
        uuid_text = str(uuid.uuid4())
        save_file_name = f"transcript-{uuid_text}.txt"
        for i, segment in enumerate(segments):
            segment_audio_file = f"segment_{i + 1}.mp3"
            segment.export(segment_audio_file, format="mp3")
            print(f"{segment_audio_file}")
            self.convert_audio_to_text(segment_audio_file, save_file_name)
            os.remove(segment_audio_file)
        return save_file_name
    
    def convert_audio_to_text(self, audio_file, save_file_name):
        print(f"Converting audio to text..file name: {audio_file}.")
        
        print("Using whisper model")
        model = whisper.load_model("base")
        text = model.transcribe(audio_file)["text"]        
        print(f"Converted audio to text successfully for file: {audio_file}.")
         # save the result to a text file
        with open(save_file_name, "a") as file:
            file.write(text)
            print("Transcript saved successfully.")
        return save_file_name
    
    def convert_audio_to_text_from_url(self, url):
        #get uuid for the audio file
        uuid_audio = str(uuid.uuid4())
        save_path = f"audio-{uuid_audio}.mp3"

        self.download_mp3(url, save_path)
        path_text_file_of_audio = self.convert_audio_to_text_by_splitting(save_path)
       
        return path_text_file_of_audio

    
def transcribe_podcast_from_mp3(mp3_file):
    audio_to_text = Audio_to_Text()
    return audio_to_text.convert_audio_to_text_by_splitting(mp3_file);

def transcribe_podcast(file_url):    
    audio_to_text = Audio_to_Text()    
    # Convert the audio file to text
    path_text_file_of_audio = audio_to_text.convert_audio_to_text_from_url(file_url)
    
    # Print the result
    print(path_text_file_of_audio)
    return path_text_file_of_audio
