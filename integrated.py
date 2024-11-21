import os
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import librosa
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
# import pyAudioAnalysis.audioFeatureExtraction as aF

# Download necessary NLTK data


# Function to capture audio using sounddevice
def record_audio(filename, duration=10, samplerate=44100):
    print("Recording your answer...")
    try:
        data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16', blocking=True)
        sf.write(filename, data, samplerate, format='WAV')
        print("Recording completed and saved.")
    except Exception as e:
        print(f"Error during recording: {e}")

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand your answer. Please try again.")
        return ""
    except sr.RequestError as e:
        print(f"Error with the speech recognition service: {e}")
        return ""

# Function to capture and process audio input
def get_audio_input():
    audio_path = "user_response.wav"
    record_audio(audio_path, duration=10)
    text = transcribe_audio(audio_path)
    return text, audio_path

# Sentiment Analysis using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score

# Lexical Diversity and Word Frequency
def lexical_diversity(text):
    words = text.split()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)
    lexical_diversity_score = len(filtered_words) / len(set(filtered_words))  # Type-Token Ratio (TTR)
    return lexical_diversity_score, word_counts.most_common(5)

# Grammar Check using LanguageTool
def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

# Cognitive Complexity based on cognitive-related words
def cognitive_complexity(text):
    cognitive_words = ['analyze', 'reason', 'reflect', 'understand', 'think']
    tokens = word_tokenize(text.lower())
    cognitive_count = sum(1 for word in tokens if word in cognitive_words)
    return cognitive_count / len(tokens)  # Proportion of cognitive words

# Emotion Recognition using Hugging Face's Emotion Model
def emotion_recognition(text):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return emotion_classifier(text)

# Pause Duration Analysis using pyAudioAnalysis
def pacing_and_pause(audio_file):
    [fs, signal] = aF.read_audio_file(audio_file)
    features, _ = aF.stFeatureExtraction(signal, fs, 0.050, 0.025)
    pause_duration = np.sum(features[2] < 0.02)  # Detecting pauses based on energy threshold
    return pause_duration

# Semantic Coherence using Hugging Face's BERT embeddings
def semantic_coherence(text, question):
    similarity_model = pipeline('feature-extraction', model='bert-base-uncased')
    question_embedding = similarity_model(question)[0][0]
    text_embedding = similarity_model(text)[0][0]
    similarity_score = np.dot(question_embedding, text_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(text_embedding))
    return similarity_score

# Function to evaluate the response for multiple aspects
def evaluate_response(text, audio_file, question):
    sentiment = analyze_sentiment(text)
    lexical_div, common_words = lexical_diversity(text)
    grammar_errors = grammar_check(text)
    complexity = cognitive_complexity(text)
    emotion = emotion_recognition(text)
    pacing = pacing_and_pause(audio_file)
    coherence = semantic_coherence(text, question)

    # Combine the features into an evaluation
    evaluation = {
        "Sentiment": sentiment,
        "Lexical Diversity": lexical_div,
        "Grammar Errors": grammar_errors,
        "Cognitive Complexity": complexity,
        "Emotion": emotion,
        "Pause Duration": pacing,
        "Semantic Coherence": coherence
    }
    return evaluation

# Example: Simulate a HR Interview
def interview():
    questions = [
        "Tell me about yourself.",
        "What are your strengths?",
        "Why do you want to work here?"
    ]

    for question in questions:
        print(f"Question: {question}")
        text, audio_path = get_audio_input()

        if text:
            print(f"Transcribed Text: {text}")
            evaluation = evaluate_response(text, audio_path, question)
            print("\nEvaluation Summary:")
            for criterion, score in evaluation.items():
                print(f"{criterion}: {score}")
        else:
            print("No valid answer received. Moving to the next question.\n")

if __name__ == "__main__":
    interview()
