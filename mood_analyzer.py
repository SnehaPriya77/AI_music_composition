import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from io import BytesIO
import wave
from config import Config

class MoodAnalyzer:
    def __init__(self):
        self.setup_models()
        self.mood_embeddings = self.create_mood_embeddings()
    
    def setup_models(self):
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=Config.SENTIMENT_MODEL,
                device=0 if torch.cuda.is_available() else -1
            )
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        except Exception:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_mood_embeddings(self):
        mood_descriptions = {
            "happy": ["joyful cheerful upbeat positive energetic bright"],
            "sad": ["melancholy sorrowful depressed gloomy downcast"],
            "calm": ["peaceful tranquil serene relaxed meditative quiet"],
            "energetic": ["dynamic powerful intense vigorous exciting"],
            "mysterious": ["enigmatic dark atmospheric suspenseful eerie"],
            "romantic": ["loving tender passionate intimate gentle warm"]
        }
        embeddings = {}
        for mood, descriptions in mood_descriptions.items():
            embedding = self.embedding_model.encode(descriptions[0])
            embeddings[mood] = embedding
        return embeddings
    
    def analyze_mood(self, user_input):
        try:
            sentiment_result = self.sentiment_pipeline(user_input)[0]
            mood_category = self.classify_mood(user_input)
            energy_level = self.extract_energy_level(user_input, sentiment_result)
            parameters = self.generate_musical_parameters(
                mood_category, energy_level, sentiment_result
            )
            parameters["lyrics_snippet"] = self.generate_lyrics(mood_category)
            parameters["chord_progression"] = self.generate_chords(mood_category, parameters["key"])
            parameters["visual_theme"] = self.get_visual_theme(mood_category)
            parameters["playlist_suggestions"] = self.get_playlist_suggestions(mood_category)
            return parameters
        except Exception:
            return self.get_default_parameters()
    
    def classify_mood(self, user_input):
        input_embedding = self.embedding_model.encode([user_input])
        similarities = {}
        for mood, mood_embedding in self.mood_embeddings.items():
            similarity = cosine_similarity(
                input_embedding.reshape(1, -1),
                mood_embedding.reshape(1, -1)
            )[0][0]
            similarities[mood] = similarity
        return max(similarities, key=similarities.get)
    
    def extract_energy_level(self, text, sentiment_result):
        high_energy_words = ["energetic", "excited", "pump", "workout", "dance", "party", "fast"]
        low_energy_words = ["calm", "peaceful", "sleep", "meditate", "quiet", "soft", "slow"]
        text_lower = text.lower()
        high_energy_score = sum(1 for word in high_energy_words if word in text_lower)
        low_energy_score = sum(1 for word in low_energy_words if word in text_lower)
        if sentiment_result['label'] == 'LABEL_2':
            base_energy = 6 + (sentiment_result['score'] * 2)
        elif sentiment_result['label'] == 'LABEL_0':
            base_energy = 4 - (sentiment_result['score'] * 2)
        else:
            base_energy = 5
        energy_adjustment = (high_energy_score - low_energy_score) * 1.5
        final_energy = max(1, min(10, base_energy + energy_adjustment))
        return int(final_energy)
    
    def generate_musical_parameters(self, mood_category, energy_level, sentiment_result):
        tempo_mapping = {
            "happy": 120, "sad": 70, "calm": 80,
            "energetic": 140, "mysterious": 90, "romantic": 85
        }
        key_preference = "major" if sentiment_result['label'] == 'LABEL_2' else "minor"
        if mood_category in ["mysterious"]:
            key_preference = "minor"
        instrument_mapping = {
            "happy": ["piano", "guitar", "drums"],
            "sad": ["piano", "strings", "cello"],
            "calm": ["piano", "flute", "soft_strings"],
            "energetic": ["electric_guitar", "drums", "bass"],
            "mysterious": ["synth", "strings", "ambient_pad"],
            "romantic": ["piano", "violin", "soft_guitar"]
        }
        parameters = {
            "energy_level": energy_level,
            "tempo": int(tempo_mapping.get(mood_category, 120) + (energy_level - 5) * 5),
            "key": key_preference,
            "mood_category": mood_category,
            "instruments": instrument_mapping.get(mood_category, ["piano", "strings", "soft_synth"]),
            "time_signature": "4/4" if energy_level > 6 else "4/4",
            "genre_style": self.determine_genre(mood_category, energy_level),
            "sentiment_confidence": round(sentiment_result['score'], 2)
        }
        parameters["tempo"] = max(60, min(180, parameters["tempo"]))
        return parameters
    
    def determine_genre(self, mood, energy):
        if energy >= 8:
            return "electronic" if mood == "energetic" else "rock"
        elif energy <= 3:
            return "ambient" if mood == "calm" else "classical"
        else:
            return "folk" if mood in ["romantic", "calm"] else "jazz"
    
    def get_default_parameters(self):
        return {
            "energy_level": 5,
            "tempo": 120,
            "key": "major",
            "mood_category": "calm",
            "instruments": ["piano", "strings", "soft_synth"],
            "time_signature": "4/4",
            "genre_style": "ambient",
            "sentiment_confidence": 0.5,
            "lyrics_snippet": "Soft winds whisper through the night...",
            "chord_progression": ["C", "G", "Am", "F"],
            "visual_theme": "🌌 Calm night sky with stars",
            "playlist_suggestions": ["Weightless – Marconi Union", "River Flows in You – Yiruma"]
        }

    def generate_lyrics(self, mood):
        snippets = {
            "happy": "Dancing in the sunlight, nothing brings me down...",
            "sad": "Tears fall like rain, memories won't fade...",
            "calm": "Silent waves, drifting through the breeze...",
            "energetic": "The beat is racing, my heart won’t stop...",
            "mysterious": "Shadows whisper secrets in the dark...",
            "romantic": "Your eyes tell stories, love in every glance..."
        }
        return snippets.get(mood, "Music speaks where words fail...")

    def generate_chords(self, mood, key):
        chord_library = {
            "happy": ["C", "G", "Am", "F"],
            "sad": ["Am", "F", "C", "G"],
            "calm": ["Dm7", "G", "Cmaj7", "Am7"],
            "energetic": ["E", "B", "C#m", "A"],
            "mysterious": ["Em", "Cmaj7", "D", "Bm"],
            "romantic": ["G", "Em", "C", "D"]
        }
        chords = chord_library.get(mood, ["C", "G", "Am", "F"])
        return [chord + ("maj" if key == "major" else "m") for chord in chords]

    def get_visual_theme(self, mood):
        themes = {
            "happy": "🌞 Bright colors, sunshine, and confetti",
            "sad": "🌧️ Rainy window with soft blue tones",
            "calm": "🌌 Tranquil ocean waves under starlight",
            "energetic": "⚡ Neon lights, fast-moving visuals",
            "mysterious": "🌙 Dark forest with mist and shadows",
            "romantic": "❤️ Sunset with warm glowing tones"
        }
        return themes.get(mood, "🎵 Abstract sound waves")

    def get_playlist_suggestions(self, mood):
        playlists = {
            "happy": ["Happy – Pharrell Williams", "Good Life – OneRepublic"],
            "sad": ["Someone Like You – Adele", "Fix You – Coldplay"],
            "calm": ["Weightless – Marconi Union", "River Flows in You – Yiruma"],
            "energetic": ["Stronger – Kanye West", "Thunderstruck – AC/DC"],
            "mysterious": ["Lux Aeterna – Clint Mansell", "Time – Hans Zimmer"],
            "romantic": ["Perfect – Ed Sheeran", "All of Me – John Legend"]
        }
        return playlists.get(mood, ["Music speaks where words fail..."])

def sine(t, f): 
    return np.sin(2*np.pi*f*t)

def square(t, f): 
    return np.sign(sine(t, f))

def triangle(t, f): 
    return 2*np.arcsin(sine(t, f))/np.pi

def note_frequency(note):
    table = {
        "C":261.63,"C#":277.18,"Db":277.18,"D":293.66,"D#":311.13,"Eb":311.13,
        "E":329.63,"F":349.23,"F#":369.99,"Gb":369.99,"G":392.00,"G#":415.30,"Ab":415.30,
        "A":440.00,"A#":466.16,"Bb":466.16,"B":493.88
    }
    n = note.replace("maj","").replace("m","")
    return table.get(n, 261.63)

def chord_frequencies(symbol):
    base = symbol.replace("maj","").replace("m","")
    minor = symbol.endswith("m") and not symbol.endswith("maj")
    f0 = note_frequency(base)
    m3 = 1.18921 if minor else 1.25992
    p5 = 1.49831
    return [f0, f0*m3, f0*p5]

def synth_chord(symbol, seconds, sr, wave_kind):
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    freqs = chord_frequencies(symbol)
    if wave_kind == "sine":
        waves = [sine(t,f) for f in freqs]
    elif wave_kind == "square":
        waves = [square(t,f) for f in freqs]
    else:
        waves = [triangle(t,f) for f in freqs]
    x = np.mean(waves, axis=0)
    env = np.linspace(0,1,int(sr*0.05))
    sustain = np.ones(len(x)-2*len(env))
    release = np.linspace(1,0,len(env))
    x = np.concatenate([env, sustain, release])[:len(x)]*x
    return x

def synth_progression(chords, tempo_bpm, instrument, volume, sr=22050):
    seconds_per_beat = 60.0/tempo_bpm
    per_chord = max(1.5, 4*seconds_per_beat)
    wave_kind = "sine"
    if instrument in ["electric_guitar","guitar","soft_guitar"]: wave_kind = "triangle"
    if instrument in ["synth","ambient_pad"]: wave_kind = "square"
    parts = [synth_chord(ch, per_chord, sr, wave_kind) for ch in chords]
    x = np.concatenate(parts)
    x = x/np.max(np.abs(x)+1e-7)
    x = x * (volume/100.0)
    pcm = np.int16(x*32767)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf

st.set_page_config(page_title="AI Music", page_icon="🎵", layout="centered")
st.title("🎵 AI Mood-based Music Generator")
st.write("Describe your mood, then fine-tune the music with controls and preview the sound.")

user_input = st.text_area("Describe your mood:", "")

if "last_params" not in st.session_state:
    st.session_state.last_params = None

colA, colB = st.columns([1,1])
with colA:
    analyze_clicked = st.button("Analyze Mood")
with colB:
    st.write("")

if analyze_clicked:
    if user_input.strip():
        analyzer = MoodAnalyzer()
        params = analyzer.analyze_mood(user_input)
        st.session_state.last_params = params
    else:
        st.warning("Please enter some text to analyze.")

if st.session_state.last_params:
    p = st.session_state.last_params
    st.subheader("Detected Settings")
    st.json(p)

    st.markdown("### Adjust Your Music")
    with st.form("controls"):
        tempo = st.slider("Tempo (BPM)", 60, 180, p["tempo"])
        energy = st.slider("Energy", 1, 10, p["energy_level"])
        key_choice = st.selectbox("Key", ["major","minor"], index=0 if p["key"]=="major" else 1)
        instruments_all = ["piano","guitar","drums","strings","cello","flute","electric_guitar","bass","synth","ambient_pad","violin","soft_guitar","soft_strings"]
        chosen_instruments = st.multiselect("Instruments", instruments_all, default=p["instruments"][:3])
        genre = st.selectbox("Genre", ["electronic","rock","ambient","classical","folk","jazz"], index=["electronic","rock","ambient","classical","folk","jazz"].index(p["genre_style"]))
        volume = st.slider("Volume", 0, 100, 80)
        chord_prog = st.text_input("Chord progression (comma-separated)", ",".join(p["chord_progression"]))
        lyrics = st.text_area("Lyrics snippet", p["lyrics_snippet"])
        col1, col2, col3 = st.columns(3)
        with col1: preview = st.form_submit_button("▶️ Generate Preview")
        with col2: download_btn = st.form_submit_button("⬇️ Download Parameters JSON")
        with col3: reset_btn = st.form_submit_button("🔄 Reset to Detected")
    
    if reset_btn:
        st.session_state.last_params = p

    new_params = dict(p)
    new_params.update({
        "tempo": tempo,
        "energy_level": energy,
        "key": key_choice,
        "instruments": chosen_instruments or p["instruments"],
        "genre_style": genre,
        "lyrics_snippet": lyrics,
        "chord_progression": [c.strip() for c in chord_prog.split(",") if c.strip()]
    })
    st.session_state.last_params = new_params

    if download_btn:
        import json
        st.download_button("Download JSON", data=json.dumps(new_params, indent=2), file_name="music_params.json", mime="application/json")

    if preview:
        lead_instrument = next((i for i in new_params["instruments"] if i in ["piano","guitar","electric_guitar","synth","ambient_pad","soft_guitar"]), "piano")
        audio_buf = synth_progression(new_params["chord_progression"], new_params["tempo"], lead_instrument, volume)
        st.audio(audio_buf, format="audio/wav")
        st.success("Preview generated. Use the play/pause control to listen.")

    st.markdown("### Visual Theme & Ideas")
    st.write(new_params["visual_theme"])
    st.markdown("### Playlist Suggestions")
    for s in p["playlist_suggestions"]:
        st.write("• " + s)
else:
    st.info("Enter your mood and click “Analyze Mood”.")
