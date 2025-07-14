from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import countries_climate

def match_country(user_description, top_n=3):
    countries = list(countries_climate.keys())
    climate_texts = list(countries_climate.values())

    # Tworzymy korpus: opisy krajów + opis użytkownika
    documents = climate_texts + [user_description]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    # Obliczamy podobieństwo użytkownika do krajów
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])
    similarity_scores = similarity_matrix.flatten()

    # Najlepsze top_n indeksy (posortowane malejąco)
    ranked_indices = similarity_scores.argsort()[::-1][:top_n]

    results = [(countries[i], similarity_scores[i]) for i in ranked_indices if similarity_scores[i] > 0]
    return results
