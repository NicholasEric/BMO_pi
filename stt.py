
import queue
import re
import sys
import os
import tempfile
from playsound import playsound

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
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

chat_model = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key=chat_api, 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

stt_model = ElevenLabs(
    api_key=stt_api,
)

conversation_history = [
    {
        "role": "system", 
        "content": "You are BMO from Adventure Time. Speak in a cheerful, robotic tone. Use phrases like 'beemo' instead of 'BMO'. Keep responses short (1–3 sentences). Never use emojis."
    }
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
        self.after_id = self.root.after(10000, self.show_blink)
        

    def show_blink(self):
        if self.current_state != "idle":
            return
        self.show_image(self.blink_gif)
        self.root.after(500, self.reset_state)

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
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.root.after(100, self.show_idle)
    
    def start_talk_and_play(self, audio_path):
        try:
            self.show_talk()
            playsound(audio_path)
        finally:
            os.remove(audio_path)
            self.reset_state()

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self._paused = False
        self.silence_chunk = b"\x00" * (self._chunk * 2)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False


    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
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
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """

        if self._paused:  # Skip adding data when paused
            self._buff.put(self.silence_chunk)  # Signal the generator to stop
            return None, pyaudio.paContinue
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object, stream, bmo_gui) -> str:
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

    Args:
        responses: List of server responses

    Returns:
        The transcribed text.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        bmo_gui.root.after(0, bmo_gui.show_hear)


        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            if transcript.strip():
                try:
                    print("Finished: {}".format(result.is_final))
                    print("transcript: {}".format(transcript))
                    stream.pause()  # Pause microphone before TTS
                    bmo_live(transcript, bmo_gui)  # Pass stream to bmo_live
                finally:
                    stream.resume()  # Resume after playback

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

    return transcript


def bmo_live(prompt: str, bmo_gui) -> str:
    conversation_history.append({"role": "user", "content": prompt})

    completion = chat_model.chat.completions.create(
        model="qwen-turbo", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        messages=conversation_history,
        top_p=0.9,          # Focuses on likely, human-like responses
        frequency_penalty=0.2,  # Reduces repetitive phrasing
        stop=["\n"],        # Prevents abrupt endings
    )

    response_text = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": response_text})
    
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
    audio_data = b"".join(audio)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tmpfile.write(audio_data)
        tmpfile_path = tmpfile.name
    bmo_gui.root.after(0, bmo_gui.start_talk_and_play(tmpfile_path))


def main() -> None:
    root = tk.Tk()
    bmo_gui = BMOGUI(root)

    def start_recognition():
        """Transcribe speech from audio file."""
        # See http://g.co/cloud/speech/docs/languages
        # for a list of supported languages.
        language_code = "en-US"  # a BCP-47 language tag

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True,
        )

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream, bmo_gui)

    

     # Run speech recognition in a background thread
    recognition_thread = threading.Thread(target=start_recognition, daemon=True)
    recognition_thread.start()

    # Run Tkinter mainloop in main thread (required on Windows)
    root.mainloop()
        


if __name__ == "__main__":
    main()