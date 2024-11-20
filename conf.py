import os
import speech_recognition as sr
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util # type: ignore

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
        
        # Optionally log more information for debugging
        # print(f"Energy Mean: {energy_mean}, Energy Max: {energy_max}, Energy Scaled: {energy_scaled}")
        # print(f"Zero-Crossing Rate Mean: {zcr_mean}, Confidence Score: {confidence_score}")
        
        return round(confidence_score, 2)
    
    except Exception as e:
        print(f"Error extracting confidence: {e}")
        return 0
    
# Function to capture and transcribe audio
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please answer the question. You have 30 seconds.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            print("Processing your answer...")
            audio_path = "user_response.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            text = recognizer.recognize_google(audio)
            return text, audio_path
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your answer. Please try again.")
            return "", ""
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            return "", ""

# Main interview process
def interview():
    print("Welcome to the HR Interview Simulation!\n")
    scores = {}
    confidence_scores = {}

    for question_id, question in questions.items():
        print(f"Question {question_id}: {question}")
        user_answer, audio_path = get_audio_input()

        if not user_answer:
            print("No valid answer received. Moving to the next question...\n")
            continue

        predefined_set = predefined_answers[question_id]
        similarity_score = calculate_similarity(predefined_set, user_answer)
        confidence_score = calculate_confidence(audio_path)
        scores[question_id] = similarity_score
        confidence_scores[question_id] = confidence_score

        print(f"Your similarity score: {similarity_score}%")
        print(f"Your confidence score: {confidence_score}%\n")

    print("\nInterview Summary:")
    for question_id in questions:
        print(f"Q{question_id}: {questions[question_id]}")
        print(f" - Similarity Score: {scores.get(question_id, 'N/A')}%")
        print(f" - Confidence Score: {confidence_scores.get(question_id, 'N/A')}%")

if __name__ == "__main__":
    interview()
