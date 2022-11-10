import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Extract the length of a given message"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(len))

    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Determine whether a message starts with a verb or not"""
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenizer(sentence))
            if not pos_tags:
                continue            
            _, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def __init__(self, tokenizer=word_tokenize):
        self.tokenizer = tokenizer
        
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(self.starting_verb))
        