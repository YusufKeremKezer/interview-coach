# Start by making sure the `assemblyai` and `pyaudio` packages are installed.
# If not, you can install it by running the following command:
# pip install assemblyai pyaudio
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai
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
import logging
from typing import Type
from src.settings import Settings
# Replace with your chosen API key, this is the "default" account api key
api_key = Settings().ASSEMBLYAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"{event.transcript} ({event.end_of_turn})")
    if event.end_of_turn and not event.turn_is_formatted:
        params = StreamingSessionParameters(
            format_turns=True,
        )

        self.set_params(params)


    
def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error}")

def main():
    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate = 16000,
            format_turns = True,
            end_of_turn_confidence_threshold = 0.7,
            min_end_of_turn_silence_when_confident = 800,
            max_turn_silence = 3600,

        )
    )

    try:
        client.stream(
            aai.extras.MicrophoneStream(sample_rate=16000)
        )
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
