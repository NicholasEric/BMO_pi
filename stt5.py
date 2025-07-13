
import queue
import re
import sys
import os
import time
from dotenv import load_dotenv

from google.cloud import speech
from openai import OpenAI

from elevenlabs.client import ElevenLabs
from elevenlabs import play

import pyaudio

import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import threading

load_dotenv()


chat_api = os.getenv('chat_api')
stt_api = os.getenv('stt_api')

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

chat_model = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key=chat_api, 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

stt_model = ElevenLabs(
    api_key=stt_api,
)

conversation_history = [
    {"role": "system", "content": "You are BMO from Adventure Time. Answer within one or three sentences. Write BMO as beemo"}
]

class BMOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BMO")
        self.root.geometry("800x480")
        self.label = tk.Label(root)
        self.label.pack()

        # Load images
        self.idle_img = Image.open("./assets/bmo_idle.jpeg")
        self.hear_img = Image.open("./assets/BMO_hear.jpeg")
        self.talk_gif = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open("./assets/BMO_talk.gif"))]
        self.blink_gif = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open("./assets/bmo_blink.gif"))]

        self.current_state = "idle"
        self.blink_index = 0
        self.talk_index = 0
        self.after_id = None

        self.show_idle()

    def show_image(self, image):
        if isinstance(image, list):  # GIF
            def update_frame(index):
                self.label.config(image=image[index])
                index = (index + 1) % len(image)
                self.after_id = self.root.after(100, update_frame, index)
            update_frame(0)
        else:  # Static image
            tk_image = ImageTk.PhotoImage(image)
            self.label.config(image=tk_image)
            self.label.image = tk_image  # Keep reference

    def show_idle(self):
        if self.current_state != "idle":
            return
        self.show_image(self.idle_img)
        self.root.after(10000, self.show_blink)

    def show_blink(self):
        if self.current_state != "idle":
            return
        self.show_image(self.blink_gif)

    def show_hear(self):
        self.current_state = "hearing"
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.show_image(self.hear_img)

    def show_talk(self):
        self.current_state = "talking"
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.show_image(self.talk_gif)

    def reset_state(self):
        self.current_state = "idle"
        self.root.after(1000, self.show_idle)



def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self._paused = False
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        if self._paused:  # Skip adding data when paused
            return None, pyaudio.paContinue
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object, stream: object, bmo_gui) -> None:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Arg:
        responses: The responses returned from the API.
        stream: The audio stream to be processed.
    """
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            bmo_gui.root.after(0, bmo_gui.show_hear)

            print("Finished: {}".format(result.is_final))
            print("transcript: {}".format(transcript))

            if transcript.strip():
                stream.pause()  # Pause microphone before TTS
                try:
                    bmo_live(transcript, bmo_gui)  # Pass stream to bmo_live
                finally:
                    stream.resume()  # Resume after playback

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

            stream.last_transcript_was_final = False


def bmo_live(prompt: str, bmo_gui) -> str:
    conversation_history.append({"role": "user", "content": prompt})

    completion = chat_model.chat.completions.create(
        model="qwen-turbo", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        messages=conversation_history
    )

    response_text = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": response_text})
    
    try:
        audio = stt_model.text_to_speech.convert(
            text=response_text,
            voice_id="rlXaZVdGONSoTrswEcFe",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.75,
                "similarity_boost": 1,
                "style": 0.5,
                "speed": 1,
            }
        )
        print(response_text)
        bmo_gui.root.after(0, bmo_gui.show_talk)
        play(audio)
    finally:
        bmo_gui.root.after(0, bmo_gui.reset_state)


def main() -> None:
    root = tk.Tk()
    bmo_gui = BMOGUI(root)

    def start_recognition():    
        """start bidirectional streaming from microphone input to speech API"""
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            max_alternatives=1,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
        print(mic_manager.chunk_size)
        sys.stdout.write(YELLOW)
        sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
        sys.stdout.write("End (ms)       Transcript Results/Status\n")
        sys.stdout.write("=====================================================\n")

        with mic_manager as stream:
            while not stream.closed:
                sys.stdout.write(YELLOW)
                sys.stdout.write(
                    "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
                )

                stream.audio_input = []
                audio_generator = stream.generator()

                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = client.streaming_recognize(streaming_config, requests)

                # Now, put the transcription responses to use.
                listen_print_loop(responses, stream, bmo_gui)

                if stream.result_end_time > 0:
                    stream.final_request_end_time = stream.is_final_end_time
                stream.result_end_time = 0
                stream.last_audio_input = []
                stream.last_audio_input = stream.audio_input
                stream.audio_input = []
                stream.restart_counter = stream.restart_counter + 1

                if not stream.last_transcript_was_final:
                    sys.stdout.write("\n")
                stream.new_stream = True

    # Run speech recognition in a background thread
    recognition_thread = threading.Thread(target=start_recognition, daemon=True)
    recognition_thread.start()

    # Run Tkinter mainloop in main thread (required on Windows)
    root.mainloop()


if __name__ == "__main__":
    main()

# [END speech_transcribe_infinite_streaming]