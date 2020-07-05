import json
import plotly
import joblib
import sys
import os.path
import pandas as pd
from sqlalchemy import create_engine

import nltk
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

sys.path.append("./models/")
sys.path.append("./data/")

from process_data import process_data
from train_classifier import tokenize, StartingVerbExtractor, load_model, load_df, train_model

CATEGORIES_FILENAME = './data/categories.csv'
MESSAGES_FILENAME = './data/messages.csv'
DATABASE_FILENAME = './data/labeled_messages_db.sqlite3'
MODEL_FILENAME = "./models/classifier.pkl"


def load_dataset(messages_filename, categories_filename, database_filename):
    '''load data from the database and return X and y
    if the database is not found it creates it

    Args:
        messages_filename (str)        
        categories_filename (str)
        database_filename (str)
    
    Returns:
        df (pandas.DataFrame): dataframe containing the data

    '''
    exists = os.path.isfile(database_filename)
    if ( not exists ):
        process_data(messages_filename, categories_filename, database_filename)

    return load_df(database_filename)
        

def load_classifier(database_filename, model_filename, do_grid_search):
    '''load data from the database and return X and y
    if the model is not found it will build and train it before returing it

    Args:
        messages_filename (str)        
        model_filename (str)
        do_grid_search (bool)
    
    Returns:
        model: the trained model

    '''
    exists = os.path.isfile(model_filename)
    if ( not exists ):
        train_model(database_filename, model_filename, do_grid_search)

    return load_model(model_filename)


def create_app(df, model):
    app = Flask(__name__)

    # index webpage displays cool visuals and receives user input text for model
    @app.route('/')
    @app.route('/index')
    def index():
        
        # extract data needed for visuals
        genre_counts = df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)
        
        category_names = df.iloc[:,4:].columns
        category_boolean = (df.iloc[:,4:] != 0).sum().values
        
        
        # create visuals
        graphs = [
                # GRAPH 1 - genre graph
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts
                    )
                ],

                'layout': {
                    'title': 'Message Genres Distribution',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            },
                # GRAPH 2 - category graph    
            {
                'data': [
                    Bar(
                        x=category_names,
                        y=category_boolean
                    )
                ],

                'layout': {
                    'title': 'Categories Distribution',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Category",
                        'tickangle': 35
                    }
                }
            }
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
    
    return app


def main():
    df = load_dataset(MESSAGES_FILENAME, CATEGORIES_FILENAME, DATABASE_FILENAME)
    model = load_classifier(DATABASE_FILENAME, MODEL_FILENAME, False)
    app = create_app(df, model)
    app.run(debug=True)


if __name__ == '__main__':
    main()
