import pandas as pd
from pysentimiento import create_analyzer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Global analyzer instance untuk setiap worker process
sentiment_analyzer = None

def init_sentiment_worker():
    """Inisialisasi analyzer untuk setiap process worker."""
    global sentiment_analyzer
    sentiment_analyzer = create_analyzer(task="sentiment", lang="en")


def classify_sentiment(text, analyzer):
    if not isinstance(text, str):
        text = str(text)
    try:
        sentiment = analyzer.predict(text)
        if sentiment.output == "POS":
            return "positive", text
        elif sentiment.output == "NEG":
            return "negative", text
        else:
            return "neutral", text
    except KeyError:
        return "neutral", text


def classify_sentiment_worker(text):
    """Worker function yang menggunakan analyzer global."""
    if not isinstance(text, str):
        text = str(text)
    try:
        sentiment = sentiment_analyzer.predict(text)
        if sentiment.output == "POS":
            return "positive", text
        elif sentiment.output == "NEG":
            return "negative", text
        else:
            return "neutral", text
    except KeyError:
        return "neutral", text


def process_sentiments_chunk(data_chunk):
    """Proses klasifikasi untuk satu chunk data pada worker."""
    return [classify_sentiment_worker(text) for text in data_chunk]


def process_sentiments_multiprocessing(data, num_workers=None):
    """Jalankan klasifikasi sentimen memakai multiprocessing dengan initializer."""
    if num_workers is None:
        num_workers = cpu_count()

    data_split = np.array_split(data, num_workers)
    with Pool(num_workers, initializer=init_sentiment_worker) as pool:
        results = list(tqdm(pool.imap(process_sentiments_chunk, data_split), total=len(data_split), desc="Classifying Sentiments"))

    return [item for sublist in results for item in sublist]


def process_sentiments_sequential(data):
    analyzer = create_analyzer(task="sentiment", lang="en")
    results = []
    for text in data:
        results.append(classify_sentiment(text, analyzer))
    return results
