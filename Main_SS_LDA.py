import numpy as np
from collections import defaultdict
# import pandas as pd
# import random
# import SS_Sentiment_stopwords as SS
from tqdm import tqdm  # Impor tqdm
# import klasifikasi
# import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


class SSLDA:
    def __init__(self, num_topics, alpha, beta, iterations):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
    
    def initialize(self, sentence_segments):
        self.sentence_segments = sentence_segments
        self.topic_assignments = []  # Store topic assignments for each segment
        self.topic_word_dist = defaultdict(lambda: defaultdict(int))  # Word distributions per topic
        
        # Initialize count matrices
        self.segment_topic_count = defaultdict(int)  # N^Z_SS
        self.word_topic_count = defaultdict(lambda: defaultdict(int))  # N^Z_w
        self.topic_word_count = defaultdict(int)  # N^Z
        self.total_topic_count = defaultdict(int)  # Total number of sentence segments in each topic
        
        # Random initialization of topics
        for segment in sentence_segments:
            # print(segment)
            topic = np.random.choice(self.num_topics)
            self.topic_assignments.append(topic)
            self.segment_topic_count[topic] += 1
            for word in segment:
                self.word_topic_count[word][topic] += 1
                self.topic_word_count[topic] += 1
    
    def sample_new_topic(self, segment, current_topic):
        # Decrease counts for current topic
        self.segment_topic_count[current_topic] -= 1
        for word in segment:
            self.word_topic_count[word][current_topic] -= 1
            self.topic_word_count[current_topic] -= 1

        # Compute topic probabilities
        topic_probs = []
        for topic in range(self.num_topics):
            # Calculate left side (how common is the topic in all segments)
            topic_prob = (self.segment_topic_count[topic] + self.alpha)
            
            # Calculate right side (how common are the words in this topic)
            for word in segment:
                word_prob = (self.word_topic_count[word][topic] + self.beta) / (self.topic_word_count[topic] + len(self.word_topic_count) * self.beta)
                topic_prob *= word_prob
            
            topic_probs.append(topic_prob)

        # Check for NaN or zero probabilities
        topic_probs = np.array(topic_probs)
        if np.any(np.isnan(topic_probs)) or topic_probs.sum() == 0:
            # If NaN or zero sum, assign uniform probabilities
            topic_probs = np.ones(self.num_topics) / self.num_topics
        else:
            # Normalize probabilities to sum to 1
            topic_probs /= topic_probs.sum()

        # Sample new topic based on the calculated probabilities
        new_topic = np.random.choice(np.arange(self.num_topics), p=topic_probs)

        # Update counts for the new topic
        self.segment_topic_count[new_topic] += 1
        for word in segment:
            self.word_topic_count[word][new_topic] += 1
            self.topic_word_count[new_topic] += 1

        # print(new_topic)

        return new_topic

    def fit(self):
        for iteration in tqdm(range(self.iterations), desc="Iterations"):
            for i, segment in enumerate(tqdm(self.sentence_segments, desc="Segments", leave=False)):
                current_topic = self.topic_assignments[i]
                new_topic = self.sample_new_topic(segment, current_topic)
                self.topic_assignments[i] = new_topic
    
    
   
    def get_topic_word_distribution(self):
        """
        Returns the word distributions per topic, sorted by word count in descending order,
        in the format: topic_id: weight*"word" + ...
        """
        topic_word_dist = defaultdict(list)

        # Iterate over the word-topic counts
        for word, topic_counts in self.word_topic_count.items():
            for topic, count in topic_counts.items():
                count = int(count)  # Convert count to int if it is not
                topic_word_dist[topic].append((word, count))

        # Sort words by count within each topic
        for topic in topic_word_dist:
            total_count = self.topic_word_count[topic]  # Total word count for this topic
            
            # Initialize a list for formatted words
            formatted_words = []
            for word, count in topic_word_dist[topic]:
                weight = count / total_count if total_count > 0 else 0
                formatted_words.append((weight, word))

            topic_word_dist[topic] = sorted(formatted_words, key=lambda x: x[0], reverse=True)  # Sort by weight

        # Sort topics by their ID (to ensure consistent ordering)
        sorted_topic_word_dist = {k: topic_word_dist[k] for k in sorted(topic_word_dist)}

        # Format output to match the desired structure
        formatted_topic_word_dist = {}
        for topic, word_counts in sorted_topic_word_dist.items():
            formatted_output = " + ".join([f"{weight:.3f}*\"{word}\"" for weight, word in word_counts[:20]])  # Top 10 words
            formatted_topic_word_dist[topic] = formatted_output

        return formatted_topic_word_dist




    
    def get_segment_topic_probabilities(self):
        """
        Returns the topic probabilities for each segment.
        """
        segment_probs = []
        for segment in self.sentence_segments:
            probs = []
            for topic in range(self.num_topics):
                topic_prob = (self.segment_topic_count[topic] + self.alpha)
                for word in segment:
                    word_prob = (self.word_topic_count[word][topic] + self.beta) / (self.topic_word_count[topic] + len(self.word_topic_count) * self.beta)
                    topic_prob *= word_prob
                probs.append(topic_prob)

            # Normalize probabilities
            total_prob = sum(probs)
            probs = [prob / total_prob for prob in probs]
            segment_probs.append(probs)
            
        # print(segment_probs)
        return segment_probs
    
    def get_topic_words(self, top_n=10):
        top_words_per_topic = []
        topic_word_dist = self.get_topic_word_distribution()

        for word_counts in topic_word_dist.values():
            # Menguraikan output menjadi list of tuples
            word_counts_list = [tuple(word_count.strip().split('*')) for word_count in word_counts.split(' + ')]

            # Ambil top N words
            top_words = [word.strip('"') for _, word in word_counts_list[:top_n]]  # Ambil kata dari tuple
            top_words_per_topic.append(top_words)  # Menambahkan ke list

        return top_words_per_topic  # Mengembalikan list dari list top_words

    
    def calculate_coherence_score(self):
        """
        Calculate multiple coherence scores for the model.
        Returns a dictionary of coherence scores (c_v, u_mass, npmi, uci).
        """
        # Get top words for each topic
        topics = self.get_topic_words()  # Get topics with the correct format
        
        # Convert sentence segments into a list of words (for Gensim input)
        texts = [segment for segment in self.sentence_segments]

        # Create a Gensim dictionary
        dictionary = Dictionary(texts)

        # Calculate different coherence scores using CoherenceModel
        coherence_scores = {}

        # c_v coherence
        coherence_model_c_v = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores['c_v'] = coherence_model_c_v.get_coherence()
        # print("cv : ", coherence_scores['c_v'])

        # u_mass coherence
        coherence_model_u_mass = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_scores['u_mass'] = coherence_model_u_mass.get_coherence()
        # print("u_mass: ", coherence_scores['u_mass'])

        # npmi coherence
        coherence_model_npmi = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_npmi')
        coherence_scores['npmi'] = coherence_model_npmi.get_coherence()
        # print("npmi: ",coherence_scores['npmi'])

        # uci coherence
        coherence_model_uci = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_uci')
        coherence_scores['uci'] = coherence_model_uci.get_coherence()
        # print("uci: ",coherence_scores['uci'])

        return coherence_scores  # Return all coherence scores as a dictionary
    
    def calculate_perplexity(self):
        """
        Calculates the perplexity of the model on the sentence segments.
        """
        log_likelihood = 0
        total_words = 0

        # Loop through each sentence segment
        for segment in self.sentence_segments:
            segment_prob = 0
            for topic in range(self.num_topics):
                # Compute the topic probability
                topic_prob = (self.segment_topic_count[topic] + self.alpha)
                
                # Compute word probabilities given the topic
                word_prob = 1
                for word in segment:
                    word_prob *= (self.word_topic_count[word][topic] + self.beta) / (
                        self.topic_word_count[topic] + len(self.word_topic_count) * self.beta)
                
                # Multiply topic and word probabilities
                segment_prob += topic_prob * word_prob
            
            # Add log probability of the segment
            if segment_prob > 0:  # Avoid log of zero
                log_likelihood += np.log(segment_prob)
            total_words += len(segment)

        # Compute perplexity
        perplexity = np.exp(-log_likelihood / total_words) if total_words > 0 else float('inf')
        return perplexity
    
    def calculate_perplexity_likelihood(self):
        """
        Calculates the log perplexity similar to the method used by Gensim.
        This includes computing the likelihood of the words in the documents 
        given the topic distribution and word distribution.
        """
        log_likelihood = 0
        total_words = 0

        # Loop through each sentence segment (bukan corpus)
        for segment in self.sentence_segments:
            doc_log_likelihood = 0
            total_words_in_doc = 0
            
            # Loop through each word in the segment
            for word in segment:
                word_log_likelihood = 0

                # Sum over all topics for each word
                for topic_id in range(self.num_topics):
                    # Calculate the probability of the word under each topic
                    word_prob = (self.word_topic_count[word][topic_id] + self.beta) / (self.topic_word_count[topic_id] + len(self.word_topic_count) * self.beta)
                    # Add the log probability of the word in the current topic
                    word_log_likelihood += word_prob
                
                # Multiply by the word count in the segment
                doc_log_likelihood += np.log(word_log_likelihood)
                total_words_in_doc += 1
            
            # Add the log likelihood of this segment to the overall log likelihood
            log_likelihood += doc_log_likelihood
            total_words += total_words_in_doc

        # Calculate log perplexity
        log_perplexity = -log_likelihood / total_words if total_words > 0 else float('inf')

        # Format the log perplexity for readability
        formatted_log_perplexity = round(log_perplexity, 5)

        return formatted_log_perplexity


def preprocess_text(sentence_segments):
    processed_segments = []
    # print(sentence_segments)

    for segment in sentence_segments:
        # Tokenize the sentence into words
        words = word_tokenize(segment)
        # Remove punctuation and stopwords
        processed_segments.append(words)
    return processed_segments
