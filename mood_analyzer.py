import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
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
            return parameters
        except Exception as e:
            print(f"Error in mood analysis: {e}")
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
            "sentiment_confidence": 0.5
        }
