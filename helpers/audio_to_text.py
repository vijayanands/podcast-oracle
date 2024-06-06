import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import requests
import uuid
from pydub import AudioSegment
import os

class Audio_to_Text:
    def __init__(self, wav2vec=False):
        if not wav2vec:
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
            print("Model whisper loaded successfully.")  
        else: 
          self.pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
          self.classifier = pipeline("text-classification")  
          print("Model wav2vec loaded successfully.")    
            
       
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
            filename = f"segment_{i + 1}.mp3"
            segment.export(filename, format="mp3")
            print(f"{filename}")
            self.convert_audio_to_text(filename, save_file_name)
            os.remove(filename)
        return save_file_name
        
    def convert_audio_to_text(self, audio_file, save_file_name):
        print("Converting audio to text...")
        
        result = self.pipe(audio_file, generate_kwargs={"language": "english"})
        print("Converted audio to text successfully.")
         # save the result to a text file
        print(result)
        with open(save_file_name, "a") as file:
            file.write(result.get('text'))
            print("Transcript saved successfully.")
        return save_file_name
    
    def convert_audio_to_text_from_url(self, url):
        #get uuid for the audio file
        uuid_audio = str(uuid.uuid4())
        save_path = f"audio-{uuid_audio}.mp3"

        self.download_mp3(url, save_path)
        path_text_file_of_audio = self.convert_audio_to_text(save_path)
       
        return path_text_file_of_audio

def main():
    # Example usage:
    url = "https://chrt.fm/track/138C95/prfx.byspotify.com/e/play.podtrac.com/npr-510310/traffic.megaphone.fm/NPR7010771664.mp3"
  
    
    audio_to_text = Audio_to_Text(wav2vec=False)
    
    
    # Convert the audio file to text

    # path_text_file_of_audio = audio_to_text.convert_audio_to_text_from_url(url)
    path_text_file_of_audio = "audio-0f985518-feec-43d9-aafb-08e82c794dd0.mp3"
    path_text_file_of_audio= audio_to_text.convert_audio_to_text_by_splitting(path_text_file_of_audio)
    
    # Print the result
    print(path_text_file_of_audio)

if __name__ == "__main__":
    main()
