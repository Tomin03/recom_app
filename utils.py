import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# Tę linię uruchom raz i najlepiej osobno, np. w osobnym skrypcie setupowym
# nltk.download('punkt')

def preprocess_text(text):
    return word_tokenize(text.lower())

def get_average_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
