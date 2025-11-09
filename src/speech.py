from IPython.display import Audio, display
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
import os
TTS_LOCATION = "global"

API_ENDPOINT = (
    f"{TTS_LOCATION}-texttospeech.googleapis.com"
    if TTS_LOCATION != "global"
    else "texttospeech.googleapis.com"
)

client = texttospeech.TextToSpeechClient(
)



MODEL = "gemini-2.5-flash-tts"  # @param ["gemini-2.5-flash-tts", "gemini-2.5-pro-tts"]

# fmt: off
VOICE = "Aoede"  # @param ["Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe", "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi"]

LANGUAGE_CODE = "en-us"  # @param ["am-et", "ar-001", "ar-eg",  "az-az",  "be-by",  "bg-bg", "bn-bd", "ca-es", "ceb-ph", "cs-cz",  "da-dk",  "de-de",  "el-gr", "en-au", "en-gb", "en-in",  "en-us",  "es-es",  "es-419", "es-mx", "es-us", "et-ee", "eu-es",  "fa-ir",  "fi-fi",  "fil-ph", "fr-fr", "fr-ca", "gl-es", "gu-in",  "hi-in",  "hr-hr",  "ht-ht",  "hu-hu", "af-za", "hy-am", "id-id",  "is-is",  "it-it",  "he-il",  "ja-jp", "jv-jv", "ka-ge", "kn-in",  "ko-kr",  "kok-in", "la-va",  "lb-lu", "lo-la", "lt-lt", "lv-lv",  "mai-in", "mg-mg",  "mk-mk",  "ml-in", "mn-mn", "mr-in", "ms-my",  "my-mm",  "nb-no",  "ne-np",  "nl-nl", "nn-no", "or-in", "pa-in",  "pl-pl",  "ps-af",  "pt-br",  "pt-pt", "ro-ro", "ru-ru", "sd-in",  "si-lk",  "sk-sk",  "sl-si",  "sq-al", "sr-rs", "sv-se", "sw-ke",  "ta-in",  "te-in",  "th-th",  "tr-tr", "uk-ua", "ur-pk", "vi-vn",  "cmn-cn", "cmn-tw"]
# fmt: on



voice = texttospeech.VoiceSelectionParams(
    name=VOICE, language_code=LANGUAGE_CODE, model_name=MODEL
)


# @title capture emotion with prompts

# fmt: off
PROMPT = "You are having a conversation with a friend. Say the following in a happy and casual way"  # @param {type: "string"}
# fmt: on
TEXT = "hahaha, i did NOT expect that. can you believe it!"  # @param {type: "string"}

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=texttospeech.SynthesisInput(text=TEXT, prompt=PROMPT),
    voice=voice,
    # Select the type of audio file you want returned
    audio_config=texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    ),
)
if __name__ == "__main__":
    # play the generated audio
    print("Playing audio...")
    display(Audio(response.audio_content))
    print("Audio content written to file 'output.mp3'")
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    print("Audio content written to file 'output.mp3'")
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    print("Audio content written to file 'output.mp3'")
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
