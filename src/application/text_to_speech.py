from kokoro import KPipeline
import sounddevice as sd
pipeline = KPipeline(lang_code='a')

async def tts(text: str):
    generator = pipeline(text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        sd.play(audio, samplerate=24000)
        sd.wait()

