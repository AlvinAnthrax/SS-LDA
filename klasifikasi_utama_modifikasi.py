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


# if __name__ == '__main__':
#     # Step 1: Load data
#     # file_path = r"C:\Kuliah\Skripsi\Final_Data\merged_prep_data.xlsx"
#     # data = pd.read_excel(file_path)  # Batasi data jika diperlukan
#     # data.dropna()
#     # data = data.sample(n=15000,random_state=42) #Ambil data random sebanyak 15.000
#     # print(data.head(10))

#     # Load alternative data 
#     # file_path_alt = r"C:\Kuliah\Skripsi\Final_Data\Processed\prep_Game_RPG.xlsx"
#     file_path_alt = r"C:\Kuliah\Skripsi\Final_Data\processed_sololv.xlsx"
#     data_alt = pd.read_excel(file_path_alt)

#     header = 'processed_content'

#     if header in data_alt.columns :
#         data_alt = data_alt[['processed_content']]
#         data_alt.dropna()
    
    
#     # Step 2: Bagi data menjadi beberapa bagian untuk multiproses
#     num_cores = cpu_count()
#     data_split = np.array_split(data_alt, num_cores)

#     # Step 3: Jalankan multiproses dengan tqdm memperbarui total data
#     with Pool(num_cores) as pool:
#         results = list(tqdm(pool.imap(process_sentiments, data_split), total=len(data_alt), desc="Classifying Sentiments"))

#     # Menggabungkan hasil dari semua proses
#     all_results = [item for sublist in results for item in sublist]

#     # Step 4: Pisahkan berdasarkan kategori sentimen
#     neutral_sentences = [text for sentiment, text in all_results if sentiment == "neutral"]
#     positive_sentences = [text for sentiment, text in all_results if sentiment == "positive"]
#     negative_sentences = [text for sentiment, text in all_results if sentiment == "negative"]

#     # Step 5: Simpan hasil ke file Excel terpisah
#     # "C:\Kuliah\Skripsi\Final_Data\Testing_Data"
#     pd.DataFrame(neutral_sentences, columns=['sentence']).to_excel("C:/Kuliah/Skripsi/Final_Data/Testing_Data/neutral_sentences_classified.xlsx", index=False)
#     pd.DataFrame(positive_sentences, columns=['sentence']).to_excel("C:/Kuliah/Skripsi/Final_Data/Testing_Data/positive_sentences_classified.xlsx", index=False)
#     pd.DataFrame(negative_sentences, columns=['sentence']).to_excel("C:/Kuliah/Skripsi/Final_Data//Testing_Data/negative_sentences_classified.xlsx", index=False)
