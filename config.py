import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify HF token is loaded
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("Hugging Face token loaded successfully")
    if len(hf_token) > 10:
        masked_token = f"{hf_token[:5]}...{hf_token[-5:]}"
        print(f"Token format: {masked_token}")
else:
    print("WARNING: No Hugging Face token found in environment variables")

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    HF_TOKEN = hf_token
    
    # Model Configuration
    WHISPER_MODEL = "large-v2"
    LLM_MODEL = "gpt-4o"
    TTS_MODEL = "eleven_multilingual_v2"
    
    # Processing
    TEMP_DIR = "./temp"
    OUTPUT_DIR = "./output"
    
    # Mandarin voices - these would be replaced with actual voice IDs from ElevenLabs
    VOICE_MAPPING = {
        "chinese_female_fast": "your_voice_id_here",
        "chinese_female_moderate": "your_voice_id_here",
        "chinese_male_fast": "your_voice_id_here",
        "chinese_male_moderate": "your_voice_id_here"
    }