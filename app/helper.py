import pandas as pd
from plotly.graph_objs import Bar
from tokenizers import tokenize_with_lemma


def count_genres(df):
    """Count the number of occurrences of genres"""
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    return genre_counts, genre_names


def count_categories(df):
    """Count the number of occurrences of categories"""
    categories = df.iloc[:, 4:]
    categories_totals = categories.sum().rename('totals').reset_index().sort_values(by='totals')    
    return categories_totals.totals, categories_totals['index']


def words_by_category(df, categories):
    """Get the most frequent words by category"""
    top_words_by_category = []
    for i, column in enumerate(list(categories)):
        messages = df.loc[df[column] > 0].sample(n=500).message.apply(tokenize_with_lemma)
        lemmatized = messages.apply(lambda x: ' '.join(x))
        word_count = lemmatized.str.split(expand=True).stack().value_counts(normalize=True).rename('count').reset_index()
        word_count = word_count.sort_values(by='count').iloc[-15:].reset_index(drop=True)        
        top_words_by_category.append(Bar(
            y = word_count['index'],
            x = word_count['count'],
            name = column,
            orientation='h',
            visible = True if i == 0 else False,
        ))
    return top_words_by_category