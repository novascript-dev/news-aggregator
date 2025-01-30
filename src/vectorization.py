from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def vectorize_text(text_data):
    """
    Vectorizing text using TF-IDF.
    - Remove portuguese stopwords
    - Limit features to 10.000
    """
    stop_words = stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer
