import whisper
import re
import librosa
from collections import Counter

# ================================
# LOAD MODEL
# ================================
whisper_model = whisper.load_model("base")


# ================================
# GET AUDIO DURATION (AUTO)
# ================================
def get_audio_duration(audio_path):
    # Correct argument usage
    return librosa.get_duration(filename=audio_path)


# ================================
# SPEECH → TEXT
# ================================
def speech_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


# ================================
# TEXT → FEATURES (ENHANCED)
# ================================
def extract_voice_features(text, duration):
    text = text.lower().strip()
    words = text.split()

    word_count = len(words)

    # ---------------------------
    # Fluency: Words per second
    # ---------------------------
    wps = word_count / duration if duration > 0 else 0
    wps = min(wps, 5)  # cap unrealistic spikes

    # ---------------------------
    # Expanded filler detection
    # ---------------------------
    fillers = [
        "um", "uh", "like", "you know", "so", "actually", "basically",
        "i think", "kind of", "sort of", "you see", "right", "okay",
        "hmm", "ah", "well", "let me think"
    ]
    filler_count = sum(len(re.findall(r'\b' + f + r'\b', text)) for f in fillers)

    # ---------------------------
    # Vocabulary richness
    # ---------------------------
    unique_words = len(set(words))
    vocab_richness = unique_words / word_count if word_count > 0 else 0

    # ---------------------------
    # Sentence clarity
    # ---------------------------
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    # ---------------------------
    # Repetition detection
    # ---------------------------
    word_freq = Counter(words)
    repeated_words = sum(1 for word, count in word_freq.items() if count > 2)

    # ---------------------------
    # Pause estimation (via punctuation)
    # ---------------------------
    pause_count = text.count('.') + text.count(',')

    # ---------------------------
    # Structure detection
    # ---------------------------
    connectors = ["first", "second", "also", "moreover", "finally", "because", "therefore"]
    structure_score = sum(1 for word in connectors if word in text)

    return {
        "word_count": word_count,
        "wps": wps,
        "filler_count": filler_count,
        "vocab_richness": vocab_richness,
        "avg_sentence_length": avg_sentence_length,
        "repeated_words": repeated_words,
        "pause_count": pause_count,
        "structure_score": structure_score
    }


# ================================
# FEATURES → SCORE (STRICT + REALISTIC)
# ================================
def compute_voice_score(features):
    score = 4.8  # start slightly below perfect (real-world assumption)

    # ---------------------------
    # FLUENCY
    # ---------------------------
    if 1.5 <= features["wps"] <= 2.2:
        pass
    elif 2.2 < features["wps"] <= 3:
        score -= 0.3
    elif features["wps"] > 3:
        score -= 0.8
    elif features["wps"] < 1:
        score -= 1.0

    # ---------------------------
    # FILLERS
    # ---------------------------
    if features["filler_count"] >= 3:
        score -= 1.0
    elif features["filler_count"] > 0:
        score -= 0.4

    # ---------------------------
    # VOCABULARY
    # ---------------------------
    if features["vocab_richness"] < 0.5:
        score -= 1.0
    elif features["vocab_richness"] < 0.65:
        score -= 0.5
    elif features["vocab_richness"] > 0.75:
        score -= 0.2  # realism penalty

    # ---------------------------
    # REPETITION
    # ---------------------------
    if features["repeated_words"] > 3:
        score -= 0.8
    elif features["repeated_words"] > 1:
        score -= 0.4

    # ---------------------------
    # SENTENCE STRUCTURE
    # ---------------------------
    if features["avg_sentence_length"] < 4:
        score -= 0.8
    elif features["avg_sentence_length"] > 16:
        score -= 1.0
    elif features["avg_sentence_length"] > 12:
        score -= 0.4

    # ---------------------------
    # CONTENT LENGTH
    # ---------------------------
    if features["word_count"] < 30:
        score -= 1.5
    elif features["word_count"] < 50:
        score -= 0.7

    # ---------------------------
    # PAUSE / FLOW
    # ---------------------------
    if features["pause_count"] < 2:
        score -= 0.3
    elif features["pause_count"] > 10:
        score -= 0.5

    # ---------------------------
    # STRUCTURE BONUS
    # ---------------------------
    if features["structure_score"] >= 2:
        score += 0.3
    elif features["structure_score"] == 0:
        score -= 0.5

    # ---------------------------
    # FINAL REALISM PENALTY
    # ---------------------------
    score -= 0.2

    # ---------------------------
    # CLAMP FINAL SCORE
    # ---------------------------
    score = max(1.5, min(score, 5.0))

    return round(score, 2)


# ================================
# COMPLETE PIPELINE
# ================================
def get_soft_skill_score(audio_path):

    duration = get_audio_duration(audio_path)
    print("⏱ Duration:", duration)

    text = speech_to_text(audio_path)
    print("📝 Transcribed Text:", text)

    features = extract_voice_features(text, duration)
    print("📊 Voice Features:", features)

    score = compute_voice_score(features)
    print("🎤 FINAL VOICE SCORE (0–5):", score)

    return score