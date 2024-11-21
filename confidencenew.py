from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample transcription
transcriptions = [
    "I am very excited about this opportunity and believe I am a perfect fit.",
    "I am not sure if I am the right person for this job.",
    "I feel confident in my abilities to succeed in this role."
]

# Analyze sentiment
for transcription in transcriptions:
    sentiment = analyzer.polarity_scores(transcription)
    confidence_score = sentiment['pos']  # Confidence inferred from positive sentiment
    print(f"Text: {transcription}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence Score (approx): {confidence_score * 100:.2f}%\n")
