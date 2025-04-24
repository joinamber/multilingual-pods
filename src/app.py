import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.transcription.transcriber import PodcastTranscriber
from src.speaker_analysis.analyzer import SpeakerAnalyzer
from src.translation.translator import PodcastTranslator

class PodcastAdapter:
    def __init__(self):
        self.transcriber = PodcastTranscriber()
        self.speaker_analyzer = SpeakerAnalyzer()
        self.translator = PodcastTranslator()
        
        # Create necessary directories
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def process_podcast(self, audio_path, podcast_info=None):
        """Process a podcast from English to Mandarin"""
        print(f"Starting processing for {audio_path}")
        
        # Step 1: Transcribe the podcast
        print("Transcribing audio...")
        transcript = self.transcriber.transcribe(audio_path)
        print(f"Transcription complete. Found {len(transcript)} segments.")
        
        # Step 2: Analyze speakers
        print("Analyzing speakers...")
        speaker_data = self.speaker_analyzer.analyze_speakers(audio_path, transcript)
        print(f"Speaker analysis complete. Found {len(speaker_data)} speakers.")
        
        # Step 3: Translate to Mandarin
        print("Translating content...")
        translated_segments = self.translator.translate_to_mandarin(
            transcript, speaker_data, podcast_info
        )
        print(f"Translation complete for {len(translated_segments)} segments.")
        
        # Step 4-7: To be implemented in future modules
        print("Next steps: TTS, synchronization, and final production")
        
        return {
            "transcript": transcript,
            "speaker_data": speaker_data,
            "translated_segments": translated_segments
        }

if __name__ == "__main__":
    # For testing purposes
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        adapter = PodcastAdapter()
        result = adapter.process_podcast(audio_path, {
            "title": "Test Podcast",
            "description": "A test podcast about technology"
        })
        print("Processing complete!")
