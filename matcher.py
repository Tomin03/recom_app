# matcher.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import countries_climate


def match_country(user_description, top_n=3):
    countries = list(countries_climate.keys())
    climate_texts = list(countries_climate.values())

    documents = climate_texts + [user_description]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])

    similarity_scores = similarity_matrix.flatten()
    ranked_indices = similarity_scores.argsort()[::-1][:top_n]

    results = [(countries[i], similarity_scores[i]) for i in ranked_indices]
    return results
