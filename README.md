
# HR Interview Simulation System

A Python-based **HR Interview Simulation System** that evaluates candidates' responses based on **semantic similarity** and **confidence scoring** using audio input. This tool is designed to simulate HR interviews, providing real-time feedback on the quality and clarity of a user's answers to common HR questions.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Implementation](#implementation)
- [Workflow](#workflow)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

## Introduction
The HR Interview Simulation System is an innovative project aimed at helping users prepare for HR interviews by simulating real-world scenarios. The system asks predefined HR questions, records the user's verbal responses, and evaluates them based on:
1. **Semantic Similarity**: How well the user's response aligns with predefined ideal answers.
2. **Confidence Score**: The clarity and confidence of the audio delivery.

This feedback empowers users to improve their communication and response quality.

---

## Features
- **Speech-to-Text Conversion**: Converts audio responses to text using Google Speech Recognition.
- **Semantic Evaluation**: Compares the user's response to ideal answers using a Hugging Face model.
- **Confidence Scoring**: Uses audio properties like energy and zero-crossing rate for confidence measurement.
- **Real-Time Feedback**: Provides immediate scores for similarity and confidence.

---

## Implementation

### 1. **Semantic Similarity**
   - Leverages the `SentenceTransformer` model `all-MiniLM-L6-v2` to calculate semantic similarity.
   - Predefined answers are embedded and compared with the user's response using cosine similarity.

### 2. **Confidence Scoring**
   - Audio analysis performed using `Librosa` to calculate:
     - **Energy**: Represents speech strength.
     - **Zero-Crossing Rate (ZCR)**: Measures clarity of speech.
   - These metrics are scaled dynamically to produce a confidence score.

### 3. **Speech Recognition**
   - Uses the `speech_recognition` library to record and transcribe audio responses.

---

## Workflow

1. **Initialization**: The system presents predefined HR interview questions.
2. **Audio Input**: User answers verbally, which is recorded and transcribed.
3. **Evaluation**:
   - Semantic similarity is calculated by comparing the user's response to predefined answers.
   - Confidence scoring is based on the audio's clarity and strength.
4. **Feedback**: The system provides similarity and confidence scores for each question.
5. **Summary**: A final summary of scores is presented to the user.

---

## Technologies Used
- **Python Libraries**:
  - `speech_recognition`: For speech-to-text conversion.
  - `librosa`: For audio analysis.
  - `sentence_transformers`: For semantic similarity evaluation.
  - `numpy`: For numerical operations.
- **Pretrained Models**:
  - `all-MiniLM-L6-v2`: A Hugging Face model for natural language understanding.

---

## Usage

### Prerequisites
1. Install Python (3.8 or higher).
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Program
1. Activate your virtual environment.
2. Run the script:
   ```bash
   python interview_simulation.py
   ```

3. Follow the on-screen instructions to answer the questions.

---

## Conclusion
This project serves as an effective tool for interview preparation, providing users with actionable feedback on their responses. By focusing on both semantic content and delivery, it ensures comprehensive evaluation and improvement.

---

Feel free to contribute, suggest improvements, or share your experience using this tool. Happy interviewing! 🎤✨

