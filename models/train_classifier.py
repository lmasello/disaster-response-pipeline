import nltk
import pandas as pd
import pickle
import re
import sys

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine

from custom_metrics import display_results, overall_scores, mean_f1_score
from custom_transformers import StartingVerbExtractor, TextLengthExtractor

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = database_filepath.split('/')[1].split('.')[0]
    df = pd.read_sql(table_name, engine, )
    X = df.message.copy()
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, y, list(y.columns)


def tokenize(text):
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')    
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]        


def build_model():
    feature_transformation = FeatureUnion([
        ('text_pipeline', Pipeline([
            ('bag_of_words', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),        
        ])),
        ('starting_verb', StartingVerbExtractor(tokenizer=tokenize)),
        ('txt_length', TextLengthExtractor()),
    ])
    pipeline = Pipeline([
        ('feature_transformation', feature_transformation),
        ('classifier', RandomForestClassifier(n_estimators=25))
    ])
    parameters = [
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [30],
            'classifier__min_samples_split': [2, 4],
        },            
    ]
    return GridSearchCV(
        estimator=pipeline, 
        param_grid=parameters, 
        scoring=mean_f1_score, 
        n_jobs=-1, 
        cv=3, 
        verbose=1,
    )


def evaluate_model(model, X_test, y_test, category_names):
    y_test_pred = model.predict(X_test)
    display_results(y_test, y_test_pred, category_names)
    print('Overall scores')
    overall_scores(y_test, y_test_pred)


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()