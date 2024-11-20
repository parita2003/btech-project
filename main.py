import speech_recognition as sr
from collections import Counter

# Predefined data: HR interview questions and weighted keywords
questions = {
    1: "Tell me about yourself.",
    2: "What are your strengths?",
    3: "Why do you want to work here?"
}

# Predefined answers with keywords and their weights
predefined_keywords = {
    1: {"experience": 2, "skills": 2, "background": 1, "passion": 1},
    2: {"communication": 3, "leadership": 3, "problem-solving": 2, "teamwork": 2},
    3: {"culture": 2, "growth": 3, "values": 1, "mission": 2}
}

# Preprocessing: Tokenization and cleaning
def preprocess(text):
    # Lowercase and split text into words
    return text.lower().replace(".", "").split()

# Scoring function
def calculate_score(question_id, user_answer):
    # Preprocess user answer
    user_tokens = preprocess(user_answer)
    
    # Get predefined keywords and weights
    keywords = predefined_keywords[question_id]
    
    # Count keyword matches
    match_counts = Counter(user_tokens)
    matched_score = 0
    total_possible_score = sum(keywords.values())
    
    # Calculate weighted score
    for word, weight in keywords.items():
        if word in match_counts:
            matched_score += weight  # Add weight of matched keyword
    
    # Calculate percentage score
    percentage_score = (matched_score / total_possible_score) * 100
    return round(percentage_score, 2)

# Function to capture and transcribe audio
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please answer the question.")
        try:
            audio = recognizer.listen(source, timeout=10)  # 10 seconds to answer
            print("Processing your answer...")
            text = recognizer.recognize_google(audio)  # Using Google STT
            print(f"Your Answer: {text}\n")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your answer. Please try again.")
            return ""
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            return ""

# Main interview process
def interview():
    print("Welcome to the HR Interview Simulation!\n")
    scores = {}
    
    for question_id, question in questions.items():
        print(f"Question {question_id}: {question}")
        
        # Get user audio input
        user_answer = get_audio_input()
        
        if not user_answer:
            print("No valid answer received. Moving to the next question...\n")
            continue
        
        # Calculate score
        score = calculate_score(question_id, user_answer)
        scores[question_id] = score
        print(f"Your score for this question: {score}%\n")
    
    # Final results
    print("\nInterview Summary:")
    total_score = sum(scores.values()) / len(questions) if scores else 0
    for question_id, score in scores.items():
        print(f"Q{question_id}: {questions[question_id]} - Score: {score}%")
    print(f"Overall Score: {total_score:.2f}%")

# Run the interview
if __name__ == "__main__":
    interview()
