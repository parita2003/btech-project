from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def lexical_diversity(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)
    lexical_diversity_score = len(filtered_words) / len(set(filtered_words))  # Type-Token Ratio (TTR)
    return lexical_diversity_score, word_counts.most_common(5)

# Example usage
text = "I am excited about the opportunity to work and grow."
score, common_words = lexical_diversity(text)
print(f"Lexical Diversity Score: {score}")
print(f"Most Common Words: {common_words}")
