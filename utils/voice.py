"""Voice integration module for SLM.
Provides TTS (text-to-speech) and STT (speech-to-text) hooks.
Extensible: swap backends by changing the import.
"""

import os, sys, tempfile, asyncio
from typing import Optional
from pathlib import Path

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Optional: use OpenAI-compatible TTS API
OPENAI_TTS_ENABLED = os.environ.get("OPENAI_TTS_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Optional: use Groq for faster STT
GROQ_STT_ENABLED = os.environ.get("GROQ_STT_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class VoiceEngine:
    """Cross-platform voice I/O. Extend with custom backends."""

    def __init__(self, tts_backend: str = "auto", stt_backend: str = "auto"):
        self.tts_backend = tts_backend
        self.stt_backend = stt_backend
        self._tts_engine = None
        self._initialized = False

    def _init_tts(self):
        if self._initialized:
            return
        if HAS_TTS and self.tts_backend in ("auto", "pyttsx3"):
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)
                self._tts_engine.setProperty("volume", 0.9)
                voices = self._tts_engine.getProperty("voices")
                if voices:
                    self._tts_engine.setProperty("voice", voices[0].id)
                self._initialized = True
            except Exception:
                pass

    def speak(self, text: str) -> bool:
        """Synthesize text to speech. Returns True on success."""
        if OPENAI_TTS_ENABLED and OPENAI_API_KEY:
            return self._speak_openai(text)
        self._init_tts()
        if self._tts_engine:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                return True
            except Exception:
                pass
        # Fallback: print (headless)
        print(f"[TTS] {text}")
        return False

    def _speak_openai(self, text: str) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.audio.speech.create(
                model="tts-1", voice="alloy", input=text,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            import subprocess
            subprocess.run(["start", temp_path], shell=True, capture_output=True)
            return True
        except Exception:
            return False

    def listen(self, timeout: int = 5, phrase_time: int = 10) -> Optional[str]:
        """Capture and transcribe speech. Returns text or None."""
        if not HAS_SR:
            return None
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time)

            if GROQ_STT_ENABLED and GROQ_API_KEY:
                return self._transcribe_groq(audio)
            return r.recognize_google(audio)
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except Exception:
            return None

    def _transcribe_groq(self, audio) -> Optional[str]:
        try:
            import io
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            wav_data = io.BytesIO(audio.get_wav_data())
            wav_data.name = "audio.wav"
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo", file=wav_data,
            )
            return transcription.text
        except Exception:
            return None

    def speak_async(self, text: str) -> asyncio.Future:
        return asyncio.get_event_loop().run_in_executor(None, self.speak, text)
