from transformers import pipeline

# Load emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Sample transcription
transcriptions = [
    "I am very excited about this opportunity and believe I am a perfect fit.",
    "I am not sure if I am the right person for this job.",
    "I feel confident in my abilities to succeed in this role."
]

# Analyze emotions
for transcription in transcriptions:
    emotions = emotion_classifier(transcription)
    
    # Sort emotions by score in descending order
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    # Get the top two emotions (apply a threshold if needed)
    top_emotions = [e for e in sorted_emotions if e['score'] > 0.3][:2]  # You can adjust the threshold
    
    print(f"Text: {transcription}")
    print("Top Emotions:")
    for emotion in top_emotions:
        print(f" - {emotion['label']}: {emotion['score'] * 100:.2f}%")
    print()
 