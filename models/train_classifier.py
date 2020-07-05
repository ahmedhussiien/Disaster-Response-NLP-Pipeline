# train the classifier and saves it

import numpy as np
import pandas as pd
import joblib
import argparse
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

TABLE_NAME = 'labeled_messages'
MODEL_DEFAULT_FILENAME = 'classifier.pkl'
DATABASE_DEFAULT_FILENAME = '../data/labeled_messages_db.sqlite3'


def load_data(database_filename):
    '''load data from the database and return X and y

    Args:
        database_filename (str)
    
    Returns:
        X (pandas.DataFrame): dataframe containing the features columns
        y (pandas.DataFrame): dataframe containing the target labels

    '''

    engine = create_engine('sqlite:///'+ database_filename +'.db')
    df = pd.read_sql_table(TABLE_NAME, engine)

    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = list(df.columns[4:])

    return X, y, category_names

def tokenize(text):
    '''clean and tokenize the passed text

    Args:
        text (str): string to be processed
    
    Returns:
        tokens (list): contains tokens extracted from text

    '''
    # normalize case and remove punctuation
    remove_punc_table = str.maketrans('', '', punctuation)
    text = text.translate(remove_punc_table).lower()
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''specify if the passed sentences start with a verb

    '''
    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            
            if not pos_tags: continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model(do_grid_search = False):
    '''build a classfier model

    Args:
        do_grid_search (bool): specify if the function should do grid search or not

    Returns:
        model: built model

    '''

    pipeline = Pipeline([
        
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('cvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()) )
    ])

    if (do_grid_search):
        parameters = {
        'features__text_pipeline__cvect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__cvect__ngram_range': ((1, 1), (1, 2)), 
        'clf__estimator__min_samples_leaf': [2, 5, 10],
        'clf__estimator__max_depth': [10, 50, None]
        }

        print("Searching for best parameters for the model ⌛...\n")
        cv = GridSearchCV(pipeline, parameters)
        return cv

    else:
        return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    '''print the model evaluation

    '''

    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
       print('Category: {}'.format(category_names[i]))
       print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
       print('Accuracy: {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_name):
    '''saves the passed model to a pickle file
    '''

    joblib.dump(model, model_name)

def load_model(model_name):
    '''load a scikit learn model from pickle file

    '''

    return joblib.load(model_name)


def train_model(database_filename, model_filename, do_grid_search):
    '''train a classifier 

    Args:
        database_filename (str)
        model_filename (str)
        do_grid_search (bool)
    '''
    print('\nDownloading NLTK components needed ⌛...')
    nltk.download(['punkt', 'wordnet',  'stopwords', 'averaged_perceptron_tagger'])

    print('\nLoading data from {} ⌛...'.format(database_filename))
    X, y, category_names = load_data(database_filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print('\nBuilding model 👷‍♂️...')
    model = build_model(do_grid_search)

    print('\nTraining model 🐎...')
    model.fit(X_train, y_train)

    print('\nEvaluating model 💯...')
    evaluate_model(model, X_test, y_test, category_names)

    print('\nSaving model as {} 💾...'.format(model_filename + '.plk'))
    save_model(model, model_filename)

    print('\nTrained model saved ✅')


def parse_arguments():
    '''Parse arguments to train a classifier

    Returns:
        database_filename (str)
        model_filename (str)
        do_grid_search (bool)
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Train Classifier")
    parser.add_argument('-d', '--database-filename', type = str, default = DATABASE_DEFAULT_FILENAME, help = 'Database filename of the cleaned data')
    parser.add_argument('-m', '--model-filename', type = str, default = MODEL_DEFAULT_FILENAME, help = 'Saved model filename')
    parser.add_argument('-cv', '--do-grid-search', type=bool, default = False, help = 'Perform grid search of the parameters')
    args = parser.parse_args()

    return args.database_filename, args.model_filename, args.do_grid_search



if __name__ == '__main__':
    database_filename, model_filename, do_grid_search = parse_arguments()
    train_model(database_filename, model_filename, do_grid_search)