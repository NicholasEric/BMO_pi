import queue
import re
import sys
import os
import itertools

from google.cloud import speech_v2
import google.api_core.exceptions

import pyaudio
from dotenv import load_dotenv



# --- MicrophoneStream class from above should be included here ---

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)



def main():
    """Transcribe microphone input using Google Speech-to-Text V2."""
    client = speech_v2.SpeechClient()

    # Create a recognizer
    project_id = GOOGLE_CLOUD_PROJECT  # Replace with your project ID
    recognizer_id = "bm07"  # Unique ID for your recognizer
    recognizer_path = f"projects/{project_id}/locations/global/recognizers/{recognizer_id}"

    try:
        client.get_recognizer(name=recognizer_path)
    except google.api_core.exceptions.NotFound:
        recognizer = client.create_recognizer(
            parent=f"projects/{project_id}/locations/global",
            recognizer_id=recognizer_id,
            recognizer={
                "language_codes": ["en-US"],
                "model": "telephony", # Choose a model that fits your use case
            },
        )
        print(f"Created recognizer: {recognizer}")


    streaming_config = speech_v2.StreamingRecognitionConfig(
        config=speech_v2.RecognitionConfig(
            auto_decoding_config={},
        ),
        streaming_features=speech_v2.StreamingRecognitionFeatures(
            interim_results=True,
        ),
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()

        requests = (
            speech_v2.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                audio=chunk,
            )
            for chunk in audio_generator
        )
        
        # First request needs to send the config
        config_request = speech_v2.StreamingRecognizeRequest(
            recognizer=recognizer_path,
            streaming_config=streaming_config
        )

        def request_generator():
            yield config_request
            for request in requests:
                yield request

        print("Listening... Press Ctrl+C to stop.")
        responses = client.streaming_recognize(requests=request_generator())

        for response in responses:
            for result in response.results:
                print(f"Transcript: {result.alternatives[0].transcript}")

if __name__ == "__main__":
    main()