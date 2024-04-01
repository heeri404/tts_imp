# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from transformers import AutoProcessor, BarkModel
from transformers import VitsModel, AutoTokenizer
from elevenlabs import generate, voices
from elevenlabs import set_api_key
import random
import torch
import os
import sys
import bittensor as bt

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

# Text-to-Speech Generation Using Suno Bark's Pretrained Model
class SunoBark:
    def __init__(self, model_path="suno/bark"):
        #Load the processor and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = BarkModel.from_pretrained(model_path)
        self.device = torch.device("cuda")
        self.speaker_list = ["v2/en_speaker_0","v2/en_speaker_1","v2/en_speaker_2","v2/en_speaker_3","v2/en_speaker_4","v2/en_speaker_5","v2/en_speaker_6","v2/en_speaker_7","v2/en_speaker_8","v2/en_speaker_9"]
        self.model.to(self.device)
    def generate_speech(self, text_input):
        # Process the text wit speaker
        speaker = self.speaker_list[torch.multinomial(torch.ones(len(self.speaker_list)), 1).item()]
        inputs = self.processor(text_input, voice_preset= speaker, return_tensors="pt")
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Generate audio
        speech = self.model.generate(**inputs)
        return speech
    
class EnglishTextToSpeech:
    def __init__(self, model_path="facebook/mms-tts-eng"):
        self.model = VitsModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda")
    def generate_speech(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors="pt")
        with torch.no_grad():
            speech = self.model(**inputs).waveform
        return speech
class ElevenLabsTTS:
    def __init__(self, api_key):
        self.api_key = api_key
        self.voices = voices()
    def generate_speech(self,text_input):
        selected_voice = None
        audio = None
        try:
            set_api_key(self.api_key)
            selected_voice = random.choice(self.voices)
            audio = generate(
                text = text_input,
                voice=selected_voice.voice_id,
            )
            return audio
        except Exception as e:
            print(f"An error occurred while processing the input audio in 11 Labs: {e}")

class MeloTTS:
    def _init_(self, model_path="myshell-ai/MeloTTS-English"):
        bt.logging.info(f'.................Initializing MeloTTS model')
        self._setup_environment()
        bt.logging.info(f'.................Environment setup complete')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bt.logging.info(f'.................Device: {self.device}')
        self.model = self._load_model()
        bt.logging.info(f'.................Model loaded')
        self.speaker_ids = self.model.hps.data.spk2id
        bt.logging.info(f'.................Speaker IDs: {self.speaker_ids}')

    def _setup_environment(self):
        bt.logging.info(f'.................Setting up environment for MeloTTS')
        # Check if MeloTTS directory exists in the current working directory
        melo_dir = os.path.join(os.getcwd(), 'MeloTTS')
        bt.logging.info(f'.................MeloTTS directory: {melo_dir}')
        if not os.path.isdir(melo_dir):
            os.system('git clone https://github.com/myshell-ai/MeloTTS.git')
            os.system('python -m unidic download')
            bt.logging.info(f'.................MeloTTS directory cloned')

        # Dynamically add the MeloTTS directory to sys.path
        sys.path.append(melo_dir)
        bt.logging.info(f'.................Added MeloTTS directory to sys.path: {melo_dir}')

    def _load_model(self):
        from melo.api import TTS
        return TTS(language='EN', device=str(self.device))

    def generate_speech(self, text_input):
        speech = self.model.tts_to_file(text_input, self.speaker_ids['EN-US'])
        return speech