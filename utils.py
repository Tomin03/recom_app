# utils.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def preprocess_text(text):
    """Tokenizuje i zamienia tekst na małe litery."""
    return word_tokenize(text.lower())

def get_average_vector(tokens, model):
    """Zwraca średni wektor dla listy tokenów z Word2Vec/FastText."""
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def calculate_similarity(vec1, vec2):
    """Oblicza cosine similarity pomiędzy dwoma wektorami."""
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
