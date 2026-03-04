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
import time
# Replace with your chosen API key, this is the "default" account api key
api_key = Settings().ASSEMBLYAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




stt_client = StreamingClient(
    StreamingClientOptions(
        api_key=api_key,
        api_host="streaming.assemblyai.com",
    )
)


import threading
# ... other imports remain the same

class SttClient:
    def __init__(self):
        self.mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.stt_client = StreamingClient(
            StreamingClientOptions(
                api_key=api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        # Using a Threading Event is safer for cross-thread communication
        self.mic_enabled = threading.Event()
        self.is_session_active = True


    def on_turn(self,client: Type[StreamingClient], event: TurnEvent):
        logger.info(f"{event.transcript} ({event.end_of_turn})")
        if event.end_of_turn:
            stt_message.message = event.transcript


    def on_terminated(self,client: Type[StreamingClient], event: TerminationEvent):
        print(
            f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
        )

    def on_error(self,client: Type[StreamingClient], error: StreamingError):
        print(f"Error occurred: {error}")

    def _audio_generator(self):
        """
        This generator continuously pulls from the mic but only yields 
        real audio when self.mic_enabled is set.
        """
        for chunk in self.mic_stream:
            if not self.is_session_active:
                break
            
            if self.mic_enabled.is_set():
                yield chunk
            else:
                # Send empty bytes or silence to keep the WebSocket 
                # connection from timing out without sending voice.
                yield b'\x00' * len(chunk)

    def _connect_stt(self):
        self.stt_client.on(StreamingEvents.Turn, self.on_turn)
        self.stt_client.on(StreamingEvents.Termination, self.on_terminated)
        self.stt_client.on(StreamingEvents.Error, self.on_error)

        params = StreamingParameters(
            sample_rate=16000,
            format_turns=True,
            end_of_turn_confidence_threshold=0.9,
            min_turn_silence=800,
            max_turn_silence=3600,
        )
        self.stt_client.connect(params)

    # ... on_turn, on_terminated, on_error remain the same ...

    def stream(self):
        self._connect_stt()
        try:
            # We pass the generator, not the raw mic_stream
            self.stt_client.stream(self._audio_generator())
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.stt_client.disconnect(terminate=True)

    def start_stt(self):
        thread = Thread(target=self.stream, daemon=True)
        thread.start()
        return thread

if __name__ == "__main__":
    client = SttClient()
    client.start_stt()

    # Now toggling works instantly
    print("Microphone Active")
    client.mic_enabled.set() 
    time.sleep(10)
    
    print("Microphone Muted")
    client.mic_enabled.clear()
    time.sleep(5)

 



