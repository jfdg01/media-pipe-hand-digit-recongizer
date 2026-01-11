"""
Speech recognition module for the Film Rating System.
"""

from typing import Optional
import speech_recognition as sr


class SpeechRecognizer:
    """Handles speech-to-text conversion."""
    
    def __init__(self, language: str = "es-ES"):
        self.language = language
        self.recognizer = sr.Recognizer()
    
    def listen(self, prompt: str = "🎤 Listening... Speak now!") -> Optional[str]:
        """
        Listen for speech and convert to text.
        
        Args:
            prompt: Message to display before listening.
        
        Returns:
            Recognized text or None if failed.
        """
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print(prompt)
            
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Processing speech...")
                
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"✅ Recognized: \"{text}\"")
                return text
                
            except sr.WaitTimeoutError:
                print("❌ No speech detected within timeout.")
            except sr.UnknownValueError:
                print("❌ Could not understand the audio.")
            except sr.RequestError as e:
                print(f"❌ Speech API error: {e}")
        
        return None
