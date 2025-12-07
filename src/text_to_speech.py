from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf

pipeline = KPipeline(lang_code='a')
text = '''
Thats an interesting question I love the perspective you bring to the table.
'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=12000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)