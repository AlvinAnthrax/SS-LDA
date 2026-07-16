import gensim
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize ,TweetTokenizer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import wordnet
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import contractions
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()
tqdm.pandas()

# Kata yang sebaiknya dipertahankan selama preprocessing agar tidak salah dikoreksi
protected_terms = {'gameplay'}

# Contoh dictionary slang, silakan sesuaikan atau tambah sesuai kebutuhan
slang_dict = {
    "u": "you",
    "r": "are",
    "y": "why",
    "k": "okay",
    "thx": "thanks",
    "btw": "by the way",
    "idk": "I don't know",
    "omg": "oh my god",
    "lol": "laugh out loud",
    "b4": "before",
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

negative_X_contradictory_words = {
    'but', 'no', 'not', 'none', 'neither', 'never', 'nobody', 'nothing', 'nowhere',
    'without', 'against', 'negative', 'deny', 'reject', 'refuse', 'decline', 'unhappy',
    'sad', 'miserable', 'hopeless', 'worthless', 'useless', 'futile', 'disagree',
    'oppose', 'contrary', 'contradict', 'disapprove', 'dissatisfied', 'objection',
    'unsatisfactory', 'unpleasant', 'regret', 'resent', 'lament', 'mourn', 'grieve',
    'bemoan', 'despise', 'loathe', 'detract', 'abhor', 'dread', 'fear', 'worry',
    'anxiety', 'sorrow', 'gloom', 'melancholy', 'dismay', 'disheartened', 'despair',
    'dislike', 'aversion', 'antipathy', 'hate', 'disdain', 'however', 'although',
    'though', 'even though', 'even if', 'yet', 'whereas', 'while', 'nevertheless',
    'nonetheless', 'on the other hand', 'in contrast', 'despite', 'in spite of',
    'can', 'could', 'would', 'should', 'may', 'might', 'must'
}

def get_wordnet_pos(treebank_tag):
    """Convert Treebank POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def correct_slang(text, slang_dict):
    """Correct slang words in the text using the provided slang dictionary."""
    words = text.split()
    corrected_words = [slang_dict[word] if word in slang_dict else word for word in words]
    return ' '.join(corrected_words)

def preprocess_text(text):
    # Inisialisasi Lemma
    lemmatizer = WordNetLemmatizer()
    text = text.lower() #lowerrcase
    text = re.sub(r"[^\w\s]", "", text) #menghilangkan tanda baca
    text = correct_slang(text, slang_dict) #Membenarkan slang word
    text = contractions.fix(text) #Menyederhanakan contractions
    text = re.sub(r'\s+', ' ', text).strip() #Menghilangkan spasi yang berlebihan
    text = re.sub(r'[^a-z\s]', '', text) #Menghilangkan char yang bukan alfabet
    # Spell correction (SpellChecker)
    corrected_text = []
    for word in word_tokenize(text):
        if word in protected_terms:
            corrected_text.append(word)
            continue

        corrected_word = spell.correction(word)
        if corrected_word and corrected_word != word:
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)
    #Penghilangan Stopword dengan custom Stopwords
    stop_words_custom = stop_words.difference(negative_X_contradictory_words)
    # POS tagging and lemmatization
    pos_tagged = pos_tag(corrected_text)
    lemmatized = []
    for word, tag in pos_tagged:
        if word not in stop_words_custom:
            lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            lemmatized.append(lemmatized_word)

    # Return the final processed text
    return ' '.join(lemmatized)


def filter_tags(tokens):
    """
    Filter tokens berdasarkan POS tag untuk mempertahankan kata benda, kata kerja penting, dan collocations.
    """
    tagged = pos_tag(tokens)
    filtered = [
        word for word, tag in tagged
        if tag.startswith('NN') or  # Noun
           tag.startswith('VB')     # Verb
    ]
    return filtered


def preprocess_advance_text(text):
    """
    Preprocessing lanjutan untuk membersihkan data.
    """
    additional_stopwords = {'would', 'could', 'should', 'application', 'game','get','play'}  # Contoh tambahan stopwords
    # additional_stopwords={}
    all_stopwords = stop_words.union(additional_stopwords)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in all_stopwords]
    tokens = filter_tags(tokens)
    return ' '.join(tokens)


def preprocess_data(data):
    with Pool(multiprocessing.cpu_count() - 1) as pool:
        result = list(tqdm(pool.imap(preprocess_text, data), total=len(data)))
    return result

