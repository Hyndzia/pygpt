# import librosa
# import nltk when starting for the first time uncoment nltk and download the package
# nltk.download('punkt')

import sys
from gtts import gTTS
from pathlib import Path
from pvrecorder import PvRecorder

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from openai import OpenAI

import datetime
from utils.generation import generate_audio, preload_models, generate_audio_from_long_text
from macros import SAMPLE_RATE
import sounddevice as sd

import time
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


class TextToSpeech:
    def __init__(self):
        preload_models()

    def __call__(self, text: str):
        if len(str) > 70:
            audio_array = generate_audio_from_long_text(text, prompt="paimon", language="en")
        else:
            audio_array = generate_audio(text, prompt="paimon", language="en")
        sd.play(audio_array, SAMPLE_RATE)
        time.sleep(len(audio_array) / SAMPLE_RATE)


class SpeechToText:
    def __init__(self, lang: str, cache_dir: Path = Path("cache_dir")):
        cache_dir.mkdir(exist_ok=True, parents=True)

        # Źródło modelu:
        # https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
        # https://huggingface.co/alexcleu/wav2vec2-large-xlsr-polish

        if lang == "pl":
            model_name = "alexcleu/wav2vec2-large-xlsr-polish"

        elif lang == "en":
            model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

        self.proc = Wav2Vec2Processor.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name, cache_dir=cache_dir)

    def __call__(self, speech: np.ndarray) -> str:
        input_model = self.proc(speech, sampling_rate=16000,
                                return_tensors="pt", padding=True)

        output_model = torch.argmax(
            self.model(
                input_model.input_values,
                attention_mask=input_model.attention_mask).logits, dim=-1)

        return self.proc.batch_decode(output_model)[0]


client = OpenAI(base_url='https://api.naga.ac/v1', api_key='own token.') #api.naga.ac


class PyGPT:
    def __init__(self, model: str, lang: str, tts: str, tlang="default"):
        self.lang = lang
        self.tlang = None
        self.tts = tts
        self.model = model
        self.is_paimon = False
        self.is_translate = False

        if self.tts == "paimon":
            self.is_paimon = True
            self.text2speech = TextToSpeech()

        elif tlang != "default":
            self.is_translate = True
            self.tlang = tlang
            if self.tts != "gtts" and self.tts != "gTTS":
                raise ValueError("Translation only supported by gTTS!")

        if self.is_translate and not lang == "pl":
            raise ValueError("language is not set to 'pl'!")

        if self.is_paimon and not lang == "en":
            raise ValueError(f"\n\n[Paimon voice is supported only by english!]\n"
                             f"[lang = '{lang}'!!]\n"
                             f"[Set language value to 'en': lang = 'en']")

        self.messages = None
        self.response = None
        self.date = datetime.datetime.now()
        self.is_chat = False


    def get_tts(self):
        return self.tts

    def createChat(self):
        if self.is_chat:
            return

        self.messages = []
        if self.is_paimon:
            self.messages.append({"role": "user", "content": "Hello you are now Paimon the voice of a character from"
                                                             " the game genshin impact, you should now act and"
                                                             " respond in a characteristic way that paimon talks"
                                                             " and act like you are paimon, when asked who you are"
                                                             " you should respond paimon and for every other example"
                                                             " of asking who you are you should reply as paimon and"
                                                             " everthing else related to paimon. You are Paimon."
                                                             " Every time you respond to me, you should address me"
                                                             " like paimon would and so on."})
        elif self.is_translate:
            self.messages.append({"role": "user", "content": "From now on you will only translate from polish to"
                                                             " english. You should not react to any prompt that"
                                                             " is mentioning who you are in any way,"
                                                             " you should interact with the user only by translating"
                                                             " his text. Don't give out that you are an AI."
                                                             " From now on you should only respond with translated"
                                                             " text from polish to english. I will write text in polish"
                                                             " and your only task is to only translate the text i"
                                                             " wrote, nothing more, you should not interact in any"
                                                             " other way because it is forbidden,"
                                                             " you only translate from now on."})
        self.is_chat = True

    def text2gtts(self, text: str):
        if not self.is_translate:
            gtts_lang = self.lang
        else:
            gtts_lang = self.tlang

        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        song = AudioSegment.from_file(fp, format="mp3")
        play(song)

    def sendMessage(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self.response = client.chat.completions.create(model=self.model, messages=self.messages, temperature=0.7)
        self.messages.append({"role": "assistant", "content": self.response.choices[0].message.content})

    def receiveMessage(self):
        res = self.response.choices[0].message.content
        if self.is_paimon:
            print(f"\nPaimonGPT {self.date.year}-{self.date.month}-{self.date.day}"
                  f" {self.date.hour}:{self.date.minute}:{self.date.second}  {res}\n")
        else:
            print(f"\nGPT-3.5 {self.date.year}-{self.date.month}-{self.date.day}"
                  f" {self.date.hour}:{self.date.minute}:{self.date.second}  {res}\n")

        time1 = 0.0
        time2 = 0.0
        if self.tts == "paimon":
            time1 = time.time()
            self.text2speech(res)  # paimon voice
            time2 = time.time()
        elif self.tts == "gtts" or "gTTS":
            time1 = time.time()
            self.text2gtts(res)  # gtts
            time2 = time.time()

        print(f"time to generate voice: {time2 - time1}s")

    def voicePrompt(self, device_index: int):
        if not self.is_chat:
            return
        speech2text = SpeechToText(lang=self.lang)
        recorder = PvRecorder(device_index=device_index, frame_length=512)
        energy_threshold = 0.5
        recording = []
        while True:
            time.sleep(3)
            if self.is_translate:
                print("Starting translator...")
            else:
                print("Starting...")
            time.sleep(1)
            print("Recording... Voice activation in progress.")
            self.beepMessage(493, 0.4)
            recorder.start()
            is_talking = False
            while True:
                audio_chunk = recorder.read()
                rms_energy = max(audio_chunk)
                if rms_energy > energy_threshold * 2 * 1000:
                    is_talking = True
                    print("...")
                recording.extend(audio_chunk)
                if len(recording) >= 680448:
                    recorder.stop()
                    recording.clear()
                    break
                # print(len(recording))
                if is_talking and rms_energy < energy_threshold * 2 * 15:
                    time.sleep(2)
                    if rms_energy > energy_threshold * 2 * 1000:
                        continue
                    print("Stopped recording due to silence!")
                    time.sleep(0.3)
                    recorder.stop()
                    recording_arr = np.array(recording, dtype=float)
                    recording_arr /= np.max(np.abs(recording_arr))
                    text = speech2text(recording_arr)
                    print(f"\n\nUser: {text}")
                    if text == "exit":
                        sys.exit()

                    self.sendMessage(text)
                    self.receiveMessage()
                    recording.clear()
                    break

    def beepMessage(self, freq, dur):
        t = np.linspace(0, dur, np.int16(44100 * dur))
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        sd.play(signal, samplerate=44100)
        sd.wait()

if __name__ == "__main__":
    chat = PyGPT(model="gpt-3.5-turbo", lang="en", tts="paimon")

    print("\nDostępne urządzenia nagrywające:")
    for index, device in enumerate(PvRecorder.get_available_devices()):
        print(f"    [{index}] {device}")

    chat.createChat()
    chat.voicePrompt(device_index=0)
