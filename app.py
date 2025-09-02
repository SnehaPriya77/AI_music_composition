import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

def generate_music(prompt, duration=30, output_file="output.wav"):
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    audio_values = model.generate(**inputs, max_new_tokens=duration*50)
    audio_tensor = audio_values[0, 0].cpu().numpy()
    scipy.io.wavfile.write(output_file, rate=16000, data=audio_tensor)
    return output_file

def convert_to_mp3(wav_file, mp3_file="output.mp3"):
    sound = AudioSegment.from_wav(wav_file)
    normalized = sound.apply_gain(-sound.dBFS)
    normalized[:30*1000].export(mp3_file, format="mp3")
    return mp3_file

def plot_waveform(file):
    rate, data = scipy.io.wavfile.read(file)
    plt.figure(figsize=(8, 3))
    plt.plot(np.linspace(0, len(data)/rate, num=len(data)), data)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

st.title("🎶 AI Mood Music Generator")
mood = st.text_input("Enter mood / description:", "upbeat happy guitar music")
duration = st.slider("Duration (seconds)", 5, 30, 15)

if st.button("Generate"):
    wav_file = generate_music(mood, duration=duration)
    mp3_file = convert_to_mp3(wav_file)
    st.success("Music generated successfully!")
    st.audio(mp3_file, format="audio/mp3")
    with open(mp3_file, "rb") as f:
        st.download_button("Download MP3", f, file_name="generated_music.mp3")
    plot_waveform(wav_file)
