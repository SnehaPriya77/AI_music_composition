def mood_to_music(mood: str):
    mood = mood.lower()

    if "happy" in mood or "excited" in mood:
        return {"genre": "Pop", "tempo": "Fast (120-140 BPM)", "key": "Major"}

    elif "sad" in mood or "down" in mood:
        return {"genre": "Blues", "tempo": "Slow (60-80 BPM)", "key": "Minor"}

    elif "angry" in mood or "frustrated" in mood:
        return {"genre": "Rock/Metal", "tempo": "Fast (140+ BPM)", "key": "Minor"}

    elif "calm" in mood or "relaxed" in mood:
        return {"genre": "Lo-fi/Chill", "tempo": "Medium (80-100 BPM)", "key": "Major"}

    else:
        return {"genre": "Classical", "tempo": "Medium", "key": "Neutral"}


if __name__ == "__main__":
    sample_mood = "I'm excited for my workout!"
    result = mood_to_music(sample_mood)
    print(f"Mood: {sample_mood}")
    print("Suggested Music Parameters:", result)
