import os
from dotenv import load_dotenv

import pandas as pd

from src.preprocess.preprocess import preprocess_dataframe
from src.vectorization.vectorization import vectorize_text
from src.evaluate.elbow_method import elbow_method
from src.clustering.clustering import clusterize

from src.utils import get_top_words_per_cluster

def vectorize_df(df):
    vectorized, vectorizer = vectorize_text(df['full_text'])
    return vectorized, vectorizer

def main():
    # Load config
    load_dotenv()

    # Preprocessing data
    dataframe = preprocess_dataframe(
        pd.read_csv(os.getenv('NEWS_CSV_FILE')),
        title_label=os.getenv('NEWS_TITLE_COLUMN'),
        content_label=os.getenv('NEWS_CONTENT_COLUMN')
    )

    # Processing
    vectorized_dataframe, dataframe_vectorizer = vectorize_text(dataframe['full_text'])
    best_k = elbow_method(vectorized_dataframe, 12)
    clusters, kmeans = clusterize(vectorized_dataframe, best_k)
    dataframe['cluster'] = clusters

    # Outputs
    get_top_words_per_cluster(kmeans, dataframe_vectorizer)
    print(dataframe[[os.getenv('NEWS_TITLE_COLUMN'), 'cluster']].head())

if __name__ == "__main__":
    main()
