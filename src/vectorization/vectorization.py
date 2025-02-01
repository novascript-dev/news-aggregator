from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text_data):
    """
    Vectorizing text using TF-IDF.
    - Limit features to 10.000
    """
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer
