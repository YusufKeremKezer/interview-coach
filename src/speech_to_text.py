import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import pipeline
import collections
import time
from ten_vad import TenVad
# Pipeline yükle
from together import Together

client = Together()
response = client.audio.transcribe(
    model="openai/whisper-large-v3",
    language="en",
    response_format="json",
    timestamp_granularities="segment"
)
print(response.text)

# VAD ve parametreler
sample_rate = 16000
frame_duration = 80  # ms
frame_size = int(sample_rate * frame_duration / 1000)  #  bytes (16-bit mono)
silence_duration = 1.0  # sn sessizlik sonrası dur
speech_frames = collections.deque()

vad = TenVad(hop_size=80, threshold=0.5)

def callback(indata, frames, time_info, status):
    frame = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    if vad.is_speech(frame, sample_rate):
        speech_frames.append(frame.copy())
        silence_start = None
    else:
        if len(speech_frames) > 10:  # Minimum konuşma
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration:
                # Transkribe
                audio = np.concatenate(list(speech_frames))
                write("speech.wav", sample_rate, audio)
                # NeMo transcribe expects a list of files
                # Note: This model is English-only (en). It does not support Turkish (tr).
                result = asr_model.transcribe(["speech.wav"])
                if result and len(result) > 0:
                    print("Transkripsiyon:", result[0])
                speech_frames.clear()
                silence_start = None


#def transcribe_audio(audio_path: str):
