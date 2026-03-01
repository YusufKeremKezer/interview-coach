# Start by making sure the `assemblyai` and `pyaudio` packages are installed.
# If not, you can install it by running the following command:
# pip install assemblyai pyaudio
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

from assemblyai.streaming.v3 import (
     BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import assemblyai as aai
from ..domain.models import stt_message
import logging
from typing import Type
from ..settings import Settings
from threading import Thread
# Replace with your chosen API key, this is the "default" account api key
api_key = Settings().ASSEMBLYAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"{event.transcript} ({event.end_of_turn})")
    if event.end_of_turn:
        stt_message.message = event.transcript


def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error}")

stt_client = StreamingClient(
    StreamingClientOptions(
        api_key=api_key,
        api_host="streaming.assemblyai.com",
    )
)

stt_client.on(StreamingEvents.Turn, on_turn)
stt_client.on(StreamingEvents.Termination, on_terminated)
stt_client.on(StreamingEvents.Error, on_error)

def _run_stt():
    params = StreamingParameters(
            sample_rate = 16000,
            format_turns = True,
            end_of_turn_confidence_threshold = 0.7,
            min_end_of_turn_silence_when_confident = 800,
            max_turn_silence = 3600,
        )
        
    stt_client.connect(params)
    try:
        stt_client.stream(
            aai.extras.MicrophoneStream(sample_rate=16000)
        )
    finally:
        stt_client.disconnect(terminate=True)

def start_stt():
    thread = Thread(target=_run_stt, daemon=True)
    thread.start()
    return thread