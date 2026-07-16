import os
import joblib
import logging
from typing import Any, Dict, List, Optional
from Main_SS_LDA import SSLDA, preprocess_text as prep
from SS_Sentiment_stopwords import segment_sentence
from klasifikasi_utama_modifikasi import process_sentiments_multiprocessing, process_sentiments_sequential
from preprocess_data import preprocess_data, preprocess_advance_text
from database import load_reviews, init_db
from scraping import scrap

# Logger for pipeline
logger = logging.getLogger(__name__)

# Global review-size configuration for easier testing.
# Ubah nilai ini atau set environment variable untuk mengatur default jumlah review.
REVIEW_SAMPLE_SIZE_DEFAULT = int(os.getenv('REVIEW_SAMPLE_SIZE_DEFAULT', '25000'))
REVIEW_SAMPLE_SIZE_MIN = int(os.getenv('REVIEW_SAMPLE_SIZE_MIN', '100'))
REVIEW_SAMPLE_SIZE_MAX = int(os.getenv('REVIEW_SAMPLE_SIZE_MAX', '50000'))

ASSOCIATION_RULES_FILE = os.getenv(
    'ASSOCIATION_RULES_FILE',
    r'C:\Kuliah\Skripsi\Final_Data\Apriori\association_rules07.pkl'
)


def load_association_rules(file_path: Optional[str] = None) -> Dict[str, Any]:
    path = file_path or ASSOCIATION_RULES_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Association rules file not found: {path}")
    return joblib.load(path)


def prepare_segments(
    sentences: List[str],
    association_rules: Dict[str, Any],
    frequent_words_list: List[str],
    threshold: float = 0.5,
) -> List[str]:
    result_segments: List[str] = []
    for sentence in sentences:
        segments = segment_sentence(sentence, frequent_words_list, association_rules, threshold)
        result_segments.extend(segments)
    return result_segments


def build_topic_model(
    sentence_segments: List[List[str]],
    num_topics: int = 11,
    alpha: float = 0.04,
    beta: float = 0.08,
    iterations: int = 10,
) -> Dict[str, Any]:
    if not sentence_segments:
        return {
            'topic_word_dist': {},
            'coherence_score': {},
        }

    lda_model = SSLDA(num_topics=num_topics, alpha=alpha, beta=beta, iterations=iterations)
    lda_model.initialize(sentence_segments)
    lda_model.fit()
    return {
        'topic_word_dist': lda_model.get_topic_word_distribution(),
        'coherence_score': lda_model.calculate_coherence_score(),
    }


def analyze_package(
    package_id: str,
    sample_size: Optional[int] = None,
    use_multiprocessing: bool = True,
    num_workers: Optional[int] = None,
    association_rules_file: Optional[str] = None,
) -> Dict[str, Any]:
    # Input validation
    if not package_id or not isinstance(package_id, str):
        raise ValueError('package_id must be a non-empty string')

    try:
        requested_sample_size = sample_size if sample_size is not None else REVIEW_SAMPLE_SIZE_MIN
        target_size = min(max(requested_sample_size, REVIEW_SAMPLE_SIZE_MIN), REVIEW_SAMPLE_SIZE_MAX)

        logger.info(
            f"Starting analysis for package_id={package_id}, requested_sample_size={requested_sample_size}, "
            f"effective_sample_size={target_size}, multiprocessing={use_multiprocessing}"
        )

        # Ensure DB table exists
        logger.debug("Initializing DB table if needed")
        init_db(package_id)

        # Load reviews (may trigger scraping)
        logger.debug("Loading reviews from DB")
        content_data = load_reviews(package_id, limit=target_size)
        if not content_data:
            logger.info("No reviews found in DB, attempting to scrape")
            scrap(package_id)
            content_data = load_reviews(package_id, limit=target_size)

        if not content_data:
            raise ValueError(f'No review data available for package_id {package_id}')

        logger.info(f"Loaded {len(content_data)} raw reviews")

        # Preprocess
        effective_sample_size = min(target_size, len(content_data))
        content_data = content_data[:effective_sample_size]
        logger.debug("Running preprocess_data on content")
        content_data = preprocess_data(content_data)
        content_data = [x for x in content_data if x != '']

        if not content_data:
            raise ValueError('No valid review text after preprocessing')

        # Sentiment analysis
        logger.info("Starting sentiment analysis")
        if use_multiprocessing:
            sentiment_results = process_sentiments_multiprocessing(content_data, num_workers=num_workers)
        else:
            sentiment_results = process_sentiments_sequential(content_data)

        logger.info(f"Sentiment analysis produced {len(sentiment_results)} results")

        neutral_sentences = [text for sentiment, text in sentiment_results if sentiment == 'neutral']
        positive_sentences = [text for sentiment, text in sentiment_results if sentiment == 'positive']
        negative_sentences = [text for sentiment, text in sentiment_results if sentiment == 'negative']

        # Advance preprocessing
        neutral_sentences = [preprocess_advance_text(sentence) for sentence in neutral_sentences]
        positive_sentences = [preprocess_advance_text(sentence) for sentence in positive_sentences]
        negative_sentences = [preprocess_advance_text(sentence) for sentence in negative_sentences]

        neutral_sentences = [x for x in neutral_sentences if x != '']
        positive_sentences = [x for x in positive_sentences if x != '']
        negative_sentences = [x for x in negative_sentences if x != '']

        logger.info(f"After preprocessing: neutral={len(neutral_sentences)}, positive={len(positive_sentences)}, negative={len(negative_sentences)}")

        # Association rules
        logger.debug("Loading association rules")
        association_rules = load_association_rules(association_rules_file)
        frequent_words_set = set()
        for itemset in association_rules.get('antecedents', []):
            frequent_words_set.update(itemset)
        sorted_frequent_words_list = sorted(frequent_words_set)

        # Prepare segments and build models
        logger.info("Preparing sentence segments and building topic models")
        SS_neutral_sentence = prep(prepare_segments(neutral_sentences, association_rules, sorted_frequent_words_list))
        SS_positive_sentence = prep(prepare_segments(positive_sentences, association_rules, sorted_frequent_words_list))
        SS_negative_sentence = prep(prepare_segments(negative_sentences, association_rules, sorted_frequent_words_list))

        analysis = {
            'package_id': package_id,
            'sample_size': effective_sample_size,
            'data_counts': {
                'raw_reviews': len(content_data),
                'neutral': len(neutral_sentences),
                'positive': len(positive_sentences),
                'negative': len(negative_sentences),
            },
            'models': {
                'neutral': build_topic_model(SS_neutral_sentence),
                'positive': build_topic_model(SS_positive_sentence),
                'negative': build_topic_model(SS_negative_sentence),
            },
        }

        logger.info(f"Finished analysis for package_id={package_id}")

        return analysis

    except Exception:
        logger.exception(f"Error during analyze_package for package_id={package_id}")
        raise
