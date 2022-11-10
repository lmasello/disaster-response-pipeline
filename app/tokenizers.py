import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


def tokenize(text):
    """Tokenize a given message removing stop words"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')    
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    return [word for word in tokens if word not in stop_words]


def tokenize_with_lemma(text):
    """Tokenize a message with a lemmatizer"""
    lemmatizer = WordNetLemmatizer()
    tokens = tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens]


def tokenize_with_stem(text):
    """Tokenize a message with a stemmer"""
    stemmer = PorterStemmer()
    tokens = tokenize(text)
    return [stemmer.stem(word) for word in tokens]   