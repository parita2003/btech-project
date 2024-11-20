import os
import speech_recognition as sr
import librosa
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import tempfile
import time

# Load the Hugging Face model for semantic similarity
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Predefined data: HR interview questions and example answers
questions = {
    1: "Tell me about yourself.",
    2: "What are your strengths?",
    3: "Why do you want to work here?"
}

# Predefined answers: multiple examples for each question
predefined_answers = {
    1: [
        "I am a software developer with 5 years of experience, passionate about creating innovative solutions.",
        "I have a background in finance, with strong analytical skills and a drive to learn new technologies.",
        "I am a recent graduate in marketing, eager to apply my knowledge in a dynamic environment."
    ],
    2: [
        "My strengths include excellent communication, leadership abilities, and being a proactive team player.",
        "I excel in problem-solving, adapting to new situations, and maintaining a positive attitude under pressure.",
        "I am highly organized, detail-oriented, and skilled at managing multiple projects simultaneously."
    ],
    3: [
        "I admire this company's mission to drive innovation and would love to contribute to that vision.",
        "The opportunity to grow in a company with a strong culture and values aligns with my career goals.",
        "I am drawn to this company's emphasis on employee development and impactful projects."
    ]
}

# Function to calculate semantic similarity score
def calculate_similarity(predefined_answers, user_answer):
    predefined_embeddings = model.encode(predefined_answers, convert_to_tensor=True)
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, predefined_embeddings)
    return round(similarity_scores.max().item() * 100, 2)

# Function to calculate confidence score using Librosa
def calculate_confidence(audio_path):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute the energy (Root Mean Square energy)
        energy = librosa.feature.rms(y=y)
        energy_mean = np.mean(energy)
        
        # Compute Zero-Crossing Rate (a rough indication of speech clarity)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        
        # Normalize energy and ZCR mean
        energy_max = np.max(energy)
        energy_scaled = (energy_mean / energy_max) if energy_max != 0 else energy_mean
        
        # Dynamic confidence score scaling using energy and ZCR
        confidence_score = min(max((energy_scaled + zcr_mean) * 100, 0), 100)
        
        return round(confidence_score, 2)
    
    except Exception as e:
        print(f"Error extracting confidence: {e}")
        return 0

# Function to capture and transcribe audio
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            audio_path = "user_response.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            text = recognizer.recognize_google(audio)
            return text, audio_path
        except sr.UnknownValueError:
            return "", ""
        except sr.RequestError:
            return "", ""

# Streamlit frontend
st.title("HR Interview Simulation")

if "question_idx" not in st.session_state:
    st.session_state.question_idx = 1
    st.session_state.scores = {}
    st.session_state.confidence_scores = {}

# Display the current question
question = questions.get(st.session_state.question_idx, None)
if question:
    st.write(f"**Question {st.session_state.question_idx}:** {question}")

# Capture audio response on button click
if st.button("Start Interview"):
    if "question_idx" not in st.session_state:
        st.session_state.question_idx = 1
    else:
        st.session_state.question_idx = 1
    st.session_state.scores = {}
    st.session_state.confidence_scores = {}

# Recording button
if st.button("Record Answer"):
    with st.spinner("Recording your answer..."):
        user_answer, audio_path = get_audio_input()
        if user_answer:
            st.write(f"Your answer: {user_answer}")

            predefined_set = predefined_answers[st.session_state.question_idx]
            similarity_score = calculate_similarity(predefined_set, user_answer)
            confidence_score = calculate_confidence(audio_path)

            st.session_state.scores[st.session_state.question_idx] = similarity_score
            st.session_state.confidence_scores[st.session_state.question_idx] = confidence_score

            st.write(f"Similarity score: {similarity_score}%")
            st.write(f"Confidence score: {confidence_score}%")

# Show the next question button
if st.button("Next Question"):
    if st.session_state.question_idx < len(questions):
        st.session_state.question_idx += 1
    else:
        st.session_state.question_idx = 1  # restart or end interview

# After the interview ends, show the report table
if st.session_state.question_idx > len(questions):
    st.write("**Interview Summary:**")

    # Prepare the data for the table
    report_data = []
    avg_similarity = 0
    avg_confidence = 0

    for idx in questions:
        similarity_score = st.session_state.scores.get(idx, 0)
        confidence_score = st.session_state.confidence_scores.get(idx, 0)
        report_data.append([questions[idx], similarity_score, confidence_score])
        avg_similarity += similarity_score
        avg_confidence += confidence_score

    avg_similarity /= len(questions)
    avg_confidence /= len(questions)

    report_data.append(["**Average**", round(avg_similarity, 2), round(avg_confidence, 2)])
    
    # Show the table
    st.table(report_data)
