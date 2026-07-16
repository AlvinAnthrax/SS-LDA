import pandas as pd
import joblib
import preprocess_data as prep
from pysentimiento import create_analyzer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
import string
from multiprocessing import Pool, cpu_count
from preprocess_data import preprocess_advance_text
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Download stopwords
# nltk.download('stopwords')

# Muat stop words dalam bahasa Inggris
stop_words = set(stopwords.words('english'))

#Muat hasil train Apriori
association_rules = joblib.load('new_combined_association_rules_batches_05.pkl')

# Load Frequent Words List 
frequent_words_list = set()
for itemset in association_rules['antecedents']:
    frequent_words_list.update(itemset)

sorted_frequent_words_list = sorted(frequent_words_list)
# print(sorted_frequent_words_list)

# print(frequent_words_list)

def segment_sentence(sentence, frequent_words_list, association_rules, confidence_threshold):
    first_frequent_word_passed = False
    previous_frequent_word = None
    sentence_segments = []
    current_segment = []

    # Pisahkan kalimat menjadi kata-kata
    words = sentence.split()

    # Iterate each word in the sentence
    for word in words:
        current_segment.append(word)

        # Pengecekan kata apakah ada di frequent list
        if word in frequent_words_list:
            if first_frequent_word_passed:
                
                #Jika kata terdapat pada list maka akan di cek tingkat tresholdnya
                rule1 = association_rules[
                    (association_rules['antecedents'] == frozenset([previous_frequent_word])) &
                    (association_rules['consequents'] == frozenset([word]))
                ]
                rule2 = association_rules[
                    (association_rules['antecedents'] == frozenset([word])) &
                    (association_rules['consequents'] == frozenset([previous_frequent_word]))
                ]

                confidence1 = rule1['confidence'].values[0] if not rule1.empty else 0
                confidence2 = rule2['confidence'].values[0] if not rule2.empty else 0

                # Jika salah satu confidence di atas threshold, dilakukan segmentasi sebelum kata sekarang
                if confidence1 > confidence_threshold or confidence2 > confidence_threshold:
                    # Gabungkan kata menjadi 1 segment
                    sentence_segments.append(' '.join(current_segment[:-1])) #Menambah segment tanpa kata terakhir
                    current_segment = [word]  # Reset dengan kata sekarang
            else:
                # Pertama kali menemukan kata dalam frequent words list
                first_frequent_word_passed = True

            previous_frequent_word = word

    # Tambahkan segment terakhir
    if current_segment:
        sentence_segments.append(' '.join(current_segment))

    return sentence_segments

# Step 4: Fungsi untuk menghilangkan stop words
def remove_stopwords_from_segment(segment):
    """
    Menghapus stopwords dari sebuah string (segmen).
    Input: string (kalimat atau frasa)
    Output: string tanpa stopwords
    """
    words = segment.split()  # Memecah string menjadi list kata
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)  # Gabungkan kembali kata-kata tanpa stopwords

# Confidence threshold

# def classify_pysentimiento (text):
#     analyzer = create_analyzer(task="sentiment", lang="en")
#     if not isinstance(text, str):
#         text = str(text)  # Convert to string if not already
    
#     try:
#         sentiment =analyzer.predict(text)
#         if sentiment.output == "POS":
#             return "positive"
#         elif sentiment.output == "NEG":
#             return "negative"
#         else:
#             return "neutral"
#     except KeyError:
#         # Handle unknown words (return neutral as fallback)
#         return "neutral"

# def process_row(row_tuple):
#     index, row = row_tuple  # Unpack the tuple to get the actual row
#     sentence = row['processed_content']  # Now `sentence` refers to the content correctly
#     sentiment = classify_pysentimiento(sentence)
#     segments = segment_sentence(sentence, frequent_words_list, association_rules, confidence_threshold=0.4)
#     segments = [remove_stopwords_from_segment(segment) for segment in segments]

#     # Categorize segments by sentiment
#     return [(segment, sentiment) for segment in segments]

# if __name__ == '__main__':
#     # Step 6: Load and shuffle data
#     file_path = r"C:\Kuliah\Skripsi\New_Data\merged_prep_data.xlsx"
#     data = pd.read_excel(file_path).head(10000)

#     # Step 7: Use multiprocessing to process each row
#     with Pool(cpu_count() - 1) as pool:
#         results = list(tqdm(pool.imap(process_row, data.iterrows()), total=len(data), desc="Processing Sentences"))

#     # Step 8: Separate results by sentiment
#     neutral_sentences = []
#     positive_sentences = []
#     negative_sentences = []

#     for result in results:
#         for segment, sentiment in result:
#             if sentiment == "neutral":
#                 neutral_sentences.append(segment)
#             elif sentiment == "positive":
#                 positive_sentences.append(segment)
#             else:
#                 negative_sentences.append(segment)

#     # Step 9: Save results to Excel files
#     neutral_df = pd.DataFrame(neutral_sentences, columns=['sentence'])
#     positive_df = pd.DataFrame(positive_sentences, columns=['sentence'])
#     negative_df = pd.DataFrame(negative_sentences, columns=['sentence'])

#     neutral_df.to_excel("C:/Kuliah/Skripsi/Testing_Data/neutral_sentences_04_05.xlsx", index=False)
#     positive_df.to_excel("C:/Kuliah/Skripsi/Testing_Data/positive_sentences_04_05.xlsx", index=False)
#     negative_df.to_excel("C:/Kuliah/Skripsi/Testing_Data/negative_sentences_04_05.xlsx", index=False)