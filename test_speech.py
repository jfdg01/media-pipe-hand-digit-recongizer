"""
Simple Speech-to-Text Test Script
Uses Google Web Speech API (free, no API key required)
Requires: pip install SpeechRecognition pyaudio
"""

import speech_recognition as sr

# Language code for Spanish (Spain). Other options:
# "es-MX" = Mexico, "es-AR" = Argentina, "es-CO" = Colombia
LANGUAGE = "es-ES"


def main():
    # Create a recognizer instance
    recognizer = sr.Recognizer()
    
    # List available microphones (helpful for debugging)
    print("Available microphones:")
    for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{i}] {mic_name}")
    print()
    
    # Use the default microphone
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... (please wait)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("🎤 Listening... Speak now!")
        
        try:
            # Listen for audio (with a timeout of 10 seconds)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("Processing...")
            
            # Use Google's free Web Speech API
            text = recognizer.recognize_google(audio, language=LANGUAGE)
            print(f"\n✅ You said: \"{text}\"")
            return text
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected within timeout period.")
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"❌ API request error: {e}")
    
    return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n[Returned string]: {result}")
