import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load disaster messages and categories and returns a merged dataframe with them"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """Clean the raw data and drop duplicates"""
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.slice(start=0, stop=-2)
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    return df


def save_data(df, database_filename):
    """Save the resulting dataframe into a sqlite database"""
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = database_filename.split('/')[1].split('.')[0]
    return df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()