import torch
import streamlit as st
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
import numpy as np

@st.cache_resource
def load_model():
    return MusicGen.get_pretrained("facebook/musicgen-small")

model = load_model()

def generate_music(prompt, duration=30, output_wav="output.wav", output_mp3="output.mp3"):
    model.set_generation_params(duration=duration)
    audio = model.generate([prompt])
    audio_np = audio[0].cpu().numpy()
    audio_np = audio_np / np.max(np.abs(audio_np))
    audio_write("output", audio_np, model.sample_rate, format="wav", strategy="loudness")
    sound = AudioSegment.from_wav("output.wav")
    sound.export(output_mp3, format="mp3")
    return output_mp3

def play_music(prompt):
    st.subheader("🎵 Generated Music")
    with st.spinner("Generating music... please wait ⏳"):
        mp3_file = generate_music(prompt)
    st.audio(mp3_file, format="audio/mp3")
    with open(mp3_file, "rb") as file:
        st.download_button("⬇️ Download MP3", file, file_name="music.mp3", mime="audio/mp3")
