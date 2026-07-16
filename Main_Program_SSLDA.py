from Main_SS_LDA import SSLDA
from Main_SS_LDA import preprocess_text as prep
import pandas as pd
import numpy as np
import os
from SS_Sentiment_stopwords import segment_sentence
from nltk import word_tokenize
from klasifikasi_utama_modifikasi import process_sentiments_multiprocessing, process_sentiments_sequential
from preprocess_data import preprocess_data
from preprocess_data import preprocess_advance_text
from tqdm import tqdm
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



    

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    token = input("Enter Google Play link or package_id: ").strip()
    from shortener import extract_package_id
    package_id = extract_package_id(token) or token  # Fallback to token if not link
    if not package_id:
        print("Invalid input.")
        exit(1)
    print(f"Using package_id: {package_id}")
    from database import load_reviews, init_db
    from scraping import scrap
    # Auto-scrape if no data
    import subprocess
    # First check DB
    init_db(package_id)
    content_data = load_reviews(package_id)
    if not content_data:
        print(f"No data for {package_id}, scraping...")
        scrap(token)
        content_data = load_reviews(package_id)
    if not content_data:
        print("Scraping failed, no data.")
        exit(1)
    print(f"Loaded {len(content_data)} rows from DB for {package_id}.")


    #Melakukan preprocesing
    content_data = content_data[:100] #Sampling

    content_data = preprocess_data(content_data)
    content_data = [x for x in content_data if x != ""] #Menghilangkan data yang kosong akibat preprocess
    print(content_data[:5])

    # Sentiment Analysis
    try:
        num_cores = os.cpu_count() or 1
    except Exception:
        num_cores = 1

    # Gunakan multiprocessing helper yang sudah diinisialisasi per worker process
    results = process_sentiments_multiprocessing(content_data, num_workers=num_cores)
    all_results = results

    content_data =""#ngosongin memory bree
    neutral_sentences = [text for sentiment, text in all_results if sentiment == "neutral"]
    positive_sentences = [text for sentiment, text in all_results if sentiment == "positive"]
    negative_sentences = [text for sentiment, text in all_results if sentiment == "negative"]

    # Penghilangan stopword
    neutral_sentences = [preprocess_advance_text(sentence) for sentence in neutral_sentences]
    positive_sentences = [preprocess_advance_text(sentence) for sentence in positive_sentences]
    negative_sentences = [preprocess_advance_text(sentence) for sentence in negative_sentences]

    neutral_sentences = [x for x in neutral_sentences if x != ""]
    positive_sentences = [x for x in positive_sentences if x != ""]
    negative_sentences = [x for x in negative_sentences if x != ""]

    print("neutral : ", neutral_sentences[:5])
    print("positive : ", positive_sentences[:5])
    print("negative : ", negative_sentences[:5])

    # Load Sentence Segment
    apriori_file =r"C:\Kuliah\Skripsi\Final_Data\Apriori\association_rules07.pkl"
    #Setup Ss
    association_rules = joblib.load(apriori_file)
    frequent_words_list = set()

    # Load freq word
    for itemset in association_rules['antecedents']:
        frequent_words_list.update(itemset)
    sorted_frequent_words_list = sorted(frequent_words_list)

    SS_neutral_sentence =[]
    SS_positive_sentence =[]
    SS_negative_sentence =[]

    for sentence in neutral_sentences:
        segment = segment_sentence(sentence,sorted_frequent_words_list,association_rules,0.5)
        SS_neutral_sentence.extend(segment)
    print(SS_neutral_sentence[:5])

    for sentence in positive_sentences:
        segment=segment_sentence(sentence,sorted_frequent_words_list,association_rules,0.5)
        SS_positive_sentence.extend(segment)
    print(SS_positive_sentence[:5])

    for sentence in negative_sentences:
       segment=segment_sentence(sentence,sorted_frequent_words_list,association_rules,0.5)
       SS_negative_sentence.extend(segment)
    print(SS_negative_sentence[:5])
    
    neutral_sentences=""
    positive_sentences=""
    negative_sentences=""


    SS_neutral_sentence =prep(SS_neutral_sentence)
    SS_positive_sentence =prep(SS_positive_sentence)
    SS_negative_sentence =prep(SS_negative_sentence)

    # Neutral
    lda_model = SSLDA(num_topics=11, alpha=0.04, beta=0.08, iterations=10)
    lda_model.initialize(SS_neutral_sentence)
    lda_model.fit()

    topic_word_dist = lda_model.get_topic_word_distribution()
    print("Topic-Word Distribution:")
    for topic, distribution in topic_word_dist.items():
            print(f"Topic {topic}: {distribution}")

    coherence_Score =lda_model.calculate_coherence_score()
    print(f'Coherence Score: {coherence_Score}')

    # Positive
    lda_model.initialize(SS_positive_sentence)
    lda_model.fit()

    topic_word_dist = lda_model.get_topic_word_distribution()
    print("Topic-Word Distribution:")
    for topic, distribution in topic_word_dist.items():
            print(f"Topic {topic}: {distribution}")
            
    coherence_Score =lda_model.calculate_coherence_score()
    print(f'Coherence Score: {coherence_Score}')

    # Negative
    lda_model.initialize(SS_negative_sentence)
    lda_model.fit()

    topic_word_dist = lda_model.get_topic_word_distribution()
    print("Topic-Word Distribution:")
    for topic, distribution in topic_word_dist.items():
            print(f"Topic {topic}: {distribution}")
            
    coherence_Score =lda_model.calculate_coherence_score()
    print(f'Coherence Score: {coherence_Score}')


