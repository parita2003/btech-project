import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def cognitive_complexity(text):
    cognitive_words = ['analyze', 'reason', 'reflect', 'understand', 'think']
    tokens = word_tokenize(text.lower())
    cognitive_count = sum(1 for word in tokens if word in cognitive_words)
    return cognitive_count / len(tokens)  # Proportion of cognitive words

# Example usage
text = "I reflect on the complexities of the problem and analyze possible solutions."
complexity_score = cognitive_complexity(text)
print(f"Cognitive Complexity Score: {complexity_score}")
