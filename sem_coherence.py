from transformers import pipeline

def semantic_coherence(text, question):
    similarity_model = pipeline('feature-extraction', model='bert-base-uncased')
    question_embedding = similarity_model(question)[0][0]
    text_embedding = similarity_model(text)[0][0]
    similarity_score = np.dot(question_embedding, text_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(text_embedding))
    return similarity_score

# Example usage
text = "I believe I have the skills and experience necessary for this role."
question = "Why do you want to work here?"
coherence_score = semantic_coherence(text, question)
print(f"Semantic Coherence Score: {coherence_score}")
