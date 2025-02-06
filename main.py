import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import requests
import tempfile
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging
import base64
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    language: str = "en"  # Changed from en-US to en
    energy_threshold: int = 4000
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8

@dataclass
class LLMConfig:
    # model: str = "deepseek-r1:8b"
    model: str = "llama3.1:latest"
    api_url: str = "http://localhost:11434/api/generate"
    max_tokens: int = 1000
    timeout: int = 30

def autoplay_audio(file_path: str) -> None:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    # HTML with both autoplay and controls
    audio_html = f"""
        <audio autoplay="true" controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = config.energy_threshold
        self.recognizer.dynamic_energy_threshold = config.dynamic_energy_threshold
        self.recognizer.pause_threshold = config.pause_threshold
    
    def get_available_microphones(self) -> List[str]:
        try:
            mics = sr.Microphone.list_microphone_names()
            return mics if mics else ["Default Microphone"]
        except Exception as e:
            logger.error(f"Error listing microphones: {e}")
            return ["Default Microphone"]

    def record_audio(self, device_index: Optional[int] = None) -> Optional[sr.AudioData]:
        try:
            mic_kwargs = {"device_index": device_index} if device_index is not None else {}
            with sr.Microphone(**mic_kwargs) as source:
                st.info("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                st.info("Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                return audio
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except Exception as e:
            st.error(f"Error recording audio: {str(e)}")
            logger.error(f"Error recording audio: {e}")
            return None

    def transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        try:
            text = self.recognizer.recognize_google(audio, language=self.config.language)
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please speak clearly and try again.")
            return None
        except sr.RequestError as e:
            st.error("Could not request results from speech recognition service.")
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            logger.error(f"Error transcribing audio: {e}")
            return None

class LLMProcessor:
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def check_ollama_connection(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate_response(self, prompt: str) -> Optional[str]:
        if not self.check_ollama_connection():
            st.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            return None

        try:
            # Get current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add current time context to the prompt
            enhanced_prompt = f"Current time is {current_time}. User query: {prompt}"
            
            payload = {
                "model": self.config.model,
                "prompt": enhanced_prompt,
                "stream": False,
                "max_tokens": self.config.max_tokens
            }
            response = requests.post(
                self.config.api_url, 
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            # Clean up response format
            # Remove thinking parts
            if "<think>" in text:
                text = text.split("<think>")[-1].split("</think>")[0]
            # Remove "Assistant response:" prefix if present
            text = text.replace("Assistant response:", "").strip()
            # Normalize whitespace
            text = ' '.join(text.split())
            return text
        except requests.exceptions.Timeout:
            st.error("LLM request timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"Error generating LLM response: {str(e)}")
            logger.error(f"Error generating LLM response: {e}")
            return None

class TextToSpeech:
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def convert_to_speech(self, text: str) -> Optional[str]:
        try:
            tts = gTTS(text=text, lang=self.config.language)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                return fp.name
        except Exception as e:
            st.error(f"Error converting text to speech: {str(e)}")
            logger.error(f"Error converting text to speech: {e}")
            return None

class VoiceAssistantApp:
    def __init__(self):
        self.audio_config = AudioConfig()
        self.llm_config = LLMConfig()
        self.audio_processor = AudioProcessor(self.audio_config)
        self.llm_processor = LLMProcessor(self.llm_config)
        self.tts = TextToSpeech(self.audio_config)
        
    def initialize_session_state(self):
        if 'mic_index' not in st.session_state:
            st.session_state.mic_index = None
    
    def process_interaction(self) -> Tuple[Optional[str], Optional[str]]:
        audio_data = self.audio_processor.record_audio(st.session_state.mic_index)
        if not audio_data:
            return None, None
            
        text = self.audio_processor.transcribe_audio(audio_data)
        if not text:
            return None, None
            
        response = self.llm_processor.generate_response(text)
        if not response:
            return text, None
            
        audio_file = self.tts.convert_to_speech(response)
        if audio_file:
            try:
                autoplay_audio(audio_file)
                time.sleep(0.5)  # Small delay to ensure audio starts
            finally:
                os.unlink(audio_file)
            
        return text, response

def main():
    st.set_page_config(page_title="Voice Assistant with Ollama", layout="wide")
    st.title("Voice Assistant with Ollama Integration")
    
    app = VoiceAssistantApp()
    app.initialize_session_state()
    
    # Microphone selection
    mics = app.audio_processor.get_available_microphones()
    selected_mic = st.selectbox(
        "Select Microphone",
        options=range(len(mics)),
        format_func=lambda x: mics[x],
        key="mic_index"
    )
    
    if st.button("Start Recording", key="record_button"):
        with st.spinner("Processing..."):
            text, response = app.process_interaction()
            
            if text:
                st.write("You said:", text)
                
            if response:
                st.write("Assistant response:", response)

if __name__ == "__main__":
    main()