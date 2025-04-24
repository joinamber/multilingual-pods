from openai import OpenAI
from config import Config

class PodcastTranslator:
    def __init__(self, api_key=None, model=Config.LLM_MODEL):
        self.client = OpenAI(api_key=api_key or Config.OPENAI_API_KEY)
        self.model = model
    
    def translate_to_mandarin(self, transcript, speaker_data, podcast_info=None):
        """Translate podcast transcript to Mandarin with cultural adaptation"""
        translated_segments = []
        
        # Create context about the podcast for better translation
        podcast_context = ""
        if podcast_info:
            podcast_context = f"""
            This is from a podcast titled "{podcast_info.get('title', '')}" 
            about {podcast_info.get('description', '')}.
            """
        
        for i, segment in enumerate(transcript):
            speaker_id = segment["speaker"]
            style = speaker_data.get(speaker_id, {"tone": "neutral", "speaking_pace": "moderate"})
            
            # Context window for better translation
            prev_text = transcript[i-1]["text"] if i > 0 else ""
            current_text = segment["text"]
            next_text = transcript[i+1]["text"] if i < len(transcript)-1 else ""
            
            prompt = f"""
            {podcast_context}
            
            Translate the following English podcast segment to natural, conversational Mandarin Chinese.
            Maintain the speaker's tone which is characterized as {style['tone']} with a {style['speaking_pace']} speaking pace.
            
            Previous segment: {prev_text}
            
            SEGMENT TO TRANSLATE: {current_text}
            
            Next segment: {next_text}
            
            Requirements:
            1. Use natural, conversational Mandarin (not formal written Chinese)
            2. Adapt any cultural references, idioms, or jokes to resonate with a Chinese audience
            3. Maintain the same meaning and emotional tone as the original
            4. If there are technical terms, provide appropriate Chinese terminology
            5. Return only the translated text in Simplified Chinese characters
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert translator specializing in podcast localization from English to Mandarin Chinese."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                translated_text = response.choices[0].message.content
                
                translated_segments.append({
                    "speaker": speaker_id,
                    "start": segment["start"],
                    "end": segment["end"],
                    "original": segment["text"],
                    "translated": translated_text
                })
                
            except Exception as e:
                print(f"Error translating segment: {e}")
                # Fallback: keep original text if translation fails
                translated_segments.append({
                    "speaker": speaker_id,
                    "start": segment["start"],
                    "end": segment["end"],
                    "original": segment["text"],
                    "translated": f"[Translation error]"
                })
        
        return translated_segments
