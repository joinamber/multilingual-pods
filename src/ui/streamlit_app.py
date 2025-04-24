import os
import sys
import streamlit as st
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.app import PodcastAdapter
from config import Config

st.set_page_config(
    page_title="Multilingual Podcast Adapter",
    page_icon="🎙️",
    layout="wide"
)

def main():
    st.title("🎙️ Multilingual Podcast Adapter")
    st.write("Transform English podcasts into Mandarin with preserved tone and style.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload English Podcast File", type=["mp3", "wav", "m4a"])
    
    if uploaded_file:
        # Show audio player for the original file
        st.subheader("Original Audio")
        st.audio(uploaded_file)
        
        # Podcast details
        st.subheader("Podcast Details")
        col1, col2 = st.columns(2)
        with col1:
            podcast_title = st.text_input("Podcast Title", "My Podcast")
        with col2:
            target_language = st.selectbox("Target Language", ["Mandarin Chinese"])
        
        podcast_description = st.text_area("Brief Description (helps improve translation)", 
                                         "A conversation about technology and its impact on society.")
        
        # Processing button
        if st.button("Generate Mandarin Version"):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            try:
                with st.spinner("Processing... This may take several minutes."):
                    # Initialize the adapter
                    adapter = PodcastAdapter()
                    
                    # Create podcast info dictionary
                    podcast_info = {
                        "title": podcast_title,
                        "description": podcast_description
                    }
                    
                    # Process the podcast
                    result = adapter.process_podcast(temp_path, podcast_info)
                    
                    # Display results (for MVP, just show the translation)
                    st.subheader("Transcription and Translation Results")
                    
                    # Show a sample of the translations in a table
                    if result["translated_segments"]:
                        df_data = [
                            {
                                "Speaker": f"Speaker {s['speaker']}",
                                "English": s['original'],
                                "Mandarin": s['translated']
                            }
                            for s in result["translated_segments"][:5]  # Show first 5 segments
                        ]
                        st.write("Sample of translations (first 5 segments):")
                        st.table(df_data)
                        
                        # TODO: Add TTS and final audio output when implemented
                        st.info("Audio synthesis will be available in the next version!")
                    
            except Exception as e:
                st.error(f"Error processing podcast: {str(e)}")
            finally:
                # Clean up the temp file
                os.unlink(temp_path)

if __name__ == "__main__":
    main()
