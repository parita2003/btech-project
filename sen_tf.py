import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
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
    # Encode predefined answers and the user answer
    predefined_embeddings = model.encode(predefined_answers, convert_to_tensor=True)
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    
    # Compute similarity scores
    similarity_scores = util.pytorch_cos_sim(user_embedding, predefined_embeddings)
    best_score = similarity_scores.max().item() * 100  # Convert to percentage
    return round(best_score, 2)

# Function to capture and transcribe audio with 30-second time limit
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please answer the question. You have 30 seconds.")
        
        # Start listening with an indefinite timeout
        recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise before listening
        start_time = time.time()  # Record the start time
        audio = None
        
        while True:
            try:
                # Listen for the audio continuously with a max of 30 seconds
                audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
                # If audio is captured, break from the loop
                break
            except sr.WaitTimeoutError:
                # Handle timeout if no audio is detected in the specified period
                if time.time() - start_time >= 30:
                    print("You took too long to respond. Please try again within 30 seconds.")
                    return ""
                continue
        
        print("Processing your answer...")
        try:
            # Transcribe the audio to text
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
        
        # Get predefined answers
        predefined_set = predefined_answers[question_id]
        
        # Calculate similarity score
        score = calculate_similarity(predefined_set, user_answer)
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
