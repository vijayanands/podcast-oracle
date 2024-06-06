import whisper
import ssl

#pip install whisper 

# use the line from below to avoid verification of certificate
ssl._create_default_https_context = ssl._create_unverified_context

model = whisper.load_model("base")

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# audio_file = dataset[0]["audio"]

audio_file = "audio-0f985518-feec-43d9-aafb-08e82c794dd0.mp3"

result = model.transcribe(audio_file)

with open("caption.txt", "w") as f:
    f.write(result["text"])