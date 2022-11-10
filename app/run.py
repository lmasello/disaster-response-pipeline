import json
import joblib

import pandas as pd
import plotly

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from graph_helper import dropdown_by_category
from helper import count_genres, count_categories, words_by_category
from tokenizers import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
# Issues with pickle and large files:
# - https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts, genre_names = count_genres(df)
    cat_counts, cat_names = count_categories(df)
    top_categories = cat_names.iloc[-3:]    
    top_words_by_category = words_by_category(df, top_categories)

    # create visuals
    graphs = [
        dict(
            data=[Bar(x=genre_names,y=genre_counts)],
            layout=dict(
                 title='Distribution of Message Genres',
                 y_title='Count', 
                 x_title='Genre'
            )
        ),
        dict(
            data=[Bar(x=cat_counts, y=cat_names, orientation='h')],
            layout=dict(
                title='Number of instances of each category',
                y_title='', 
                x_title='Category', 
                margin=dict(l=120, r=10)
            ),
        ),
        dict(
            data=top_words_by_category,
            layout=dict(
                dropdown=True,
                title='Top words by category',
                y_title='', 
                x_title='Frequency',
                visible=[True, False, False],
                updatemenus=dropdown_by_category(top_categories),
            ),
        ),
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()