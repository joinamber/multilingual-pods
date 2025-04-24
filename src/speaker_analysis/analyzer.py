import librosa
import numpy as np

class SpeakerAnalyzer:
    def __init__(self):
        pass
    
    def analyze_speakers(self, audio_path, transcript):
        """Analyze speaker characteristics from audio segments"""
        speakers = {}
        audio, sr = librosa.load(audio_path)
        
        for segment in transcript:
            speaker_id = segment["speaker"]
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            
            # Skip if segment is too short
            if end_sample <= start_sample or end_sample >= len(audio):
                continue
                
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # Extract audio features
            # Note: These basic features give a rough approximation
            # In a production system, you'd use more sophisticated models
            try:
                pitch = librosa.yin(segment_audio, fmin=75, fmax=600)
                tempo, _ = librosa.beat.beat_track(y=segment_audio, sr=sr)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
                
                # Accumulate speaker data
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "samples": 1,
                        "avg_pitch": np.nanmean(pitch) if not np.all(np.isnan(pitch)) else 160,
                        "avg_tempo": tempo,
                        "avg_spectral_centroid": spectral_centroid
                    }
                else:
                    spk = speakers[speaker_id]
                    spk["samples"] += 1
                    pitch_mean = np.nanmean(pitch) if not np.all(np.isnan(pitch)) else 160
                    spk["avg_pitch"] = (spk["avg_pitch"] * (spk["samples"] - 1) + pitch_mean) / spk["samples"]
                    spk["avg_tempo"] = (spk["avg_tempo"] * (spk["samples"] - 1) + tempo) / spk["samples"]
                    spk["avg_spectral_centroid"] = (spk["avg_spectral_centroid"] * (spk["samples"] - 1) + spectral_centroid) / spk["samples"]
            except Exception as e:
                print(f"Error analyzing segment for speaker {speaker_id}: {e}")
        
        # Determine characteristics for each speaker
        for speaker_id, features in speakers.items():
            features["tone"] = "higher_pitched" if features["avg_pitch"] > 180 else "lower_pitched"
            features["speaking_pace"] = "fast" if features["avg_tempo"] > 120 else "moderate_to_slow"
        
        return speakers
