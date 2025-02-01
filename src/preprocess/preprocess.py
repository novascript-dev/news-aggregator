import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def preprocess_text(text):
    """
    Cleaning text by:
    - Turning it lowercase
    - Removing special characters
    - Removing multiple spaces
    - Removing stopwords
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('portuguese'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def preprocess_dataframe(dataframe, title_label, content_label):
    """
    Here we apply preprocess_text to the dataframe labeled field and
    put them together in a separate field labeled 'full_text'.
    """
    dataframe[content_label] = dataframe[content_label].apply(preprocess_text)
    dataframe[title_label] = dataframe[title_label].apply(preprocess_text)
    dataframe['full_text'] = dataframe[title_label] + ' ' + dataframe[content_label]
    return dataframe