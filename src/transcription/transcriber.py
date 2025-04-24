import whisperx
import os
import torch
import warnings
import torchaudio
from config import Config

# Filter out specific warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated.*")
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions.*")

# Set audio backend
torchaudio.set_audio_backend("soundfile")

class PodcastTranscriber:
    def __init__(self, model_name=Config.WHISPER_MODEL):
        self.model_name = model_name
        self.model = None
        # Use CUDA if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # Use float32 for CPU, float16 for GPU
        self.compute_type = "float32"  # Force float32 for better compatibility
        print(f"Using compute type: {self.compute_type}")
    
    def load_model(self):
        """Load the WhisperX model"""
        print(f"Loading WhisperX model: {self.model_name}")
        try:
            self.model = whisperx.load_model(
                self.model_name, 
                device=self.device,
                compute_type=self.compute_type
            )
            print("WhisperX model loaded successfully")
            return self.model
        except Exception as e:
            print(f"Error loading WhisperX model: {str(e)}")
            raise
    
    def transcribe(self, audio_path, language="en", min_speakers=2, max_speakers=2):
        """
        Transcribe audio file with speaker diarization
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (e.g., 'en' for English)
            min_speakers (int): Minimum number of speakers to detect
            max_speakers (int): Maximum number of speakers to detect
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Processing audio file: {audio_path}")
        print(f"Language specified: {language}")
        print(f"Speaker range: {min_speakers}-{max_speakers} speakers")
        
        if self.model is None:
            self.load_model()
        
        # Ensure temp directory exists
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        
        # First, get the initial transcription with specified language
        print(f"Starting transcription in {language}...")
        result = self.model.transcribe(
            audio_path,
            language=language,
            batch_size=16
        )
        print("Initial transcription completed")
        
        # Load diarization model with HF token
        print("Loading diarization model...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=Config.HF_TOKEN,
                device=self.device
            )
            print("Diarization model loaded successfully")
        except Exception as e:
            print(f"Error loading diarization model: {str(e)}")
            raise
        
        # Get speaker diarization with enhanced parameters
        print("Starting speaker diarization...")
        try:
            diarize_segments = diarize_model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            print("Speaker diarization completed")
        except Exception as e:
            print(f"Error during diarization: {str(e)}")
            raise
        
        # Assign speaker labels
        print("Assigning speaker labels...")
        try:
            result = whisperx.assign_word_speakers(
                diarize_segments, 
                result
            )
            print("Speaker labels assigned successfully")
        except Exception as e:
            print(f"Error assigning speakers: {str(e)}")
            raise
        
        # Process into structured format with timing analysis
        transcript = []
        current_speaker = None
        speaker_segments = []
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "unknown")
            
            # Track speaker changes for analysis
            if current_speaker != speaker:
                if current_speaker is not None:
                    speaker_segments.append({
                        "speaker": current_speaker,
                        "duration": segment["start"] - segment_start
                    })
                current_speaker = speaker
                segment_start = segment["start"]
            
            entry = {
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "duration": segment["end"] - segment["start"]
            }
            transcript.append(entry)
        
        # Add final speaker segment
        if current_speaker is not None:
            speaker_segments.append({
                "speaker": current_speaker,
                "duration": transcript[-1]["end"] - segment_start
            })
        
        # Analyze speaker patterns
        print("\nSpeaker Analysis:")
        speaker_stats = {}
        for segment in transcript:
            speaker = segment["speaker"]
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "count": 0,
                    "total_duration": 0,
                    "total_words": 0,
                    "segments": []
                }
            speaker_stats[speaker]["count"] += 1
            speaker_stats[speaker]["total_duration"] += segment["duration"]
            speaker_stats[speaker]["total_words"] += len(segment["text"].split())
            speaker_stats[speaker]["segments"].append(segment)
        
        print("\nDetailed Speaker Statistics:")
        total_duration = sum(stats["total_duration"] for stats in speaker_stats.values())
        
        for speaker, stats in speaker_stats.items():
            print(f"\nSpeaker {speaker}:")
            print(f"  Number of segments: {stats['count']}")
            print(f"  Total speaking time: {stats['total_duration']:.2f} seconds")
            print(f"  Speaking percentage: {(stats['total_duration']/total_duration)*100:.1f}%")
            print(f"  Total words: {stats['total_words']}")
            print(f"  Average words per segment: {stats['total_words']/stats['count']:.1f}")
            print(f"  Average segment duration: {stats['total_duration']/stats['count']:.2f} seconds")
        
        print(f"\nTranscription completed. Found {len(transcript)} segments.")
        return transcript
