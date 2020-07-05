# load, clean and save the datasets

import pandas as pd
from sqlalchemy import create_engine
import argparse

CATEGORIES_DEFAULT_FILENAME = 'categories.csv'
MESSAGES_DEFAULT_FILENAME = 'messages.csv'
DATABASE_DEFAULT_FILENAME = 'labeled_messages_db.sqlite3'
TABLE_NAME = 'labeled_messages'


def load_data(messages_filename, categories_filename):
    '''
    Load the data from the input files

    Args:
        categories_filename (str):  categories filename
        messages_filename (str):    messages filename
    Returns:
        df (pandas.DataFrame): dataframe containing the two datasets merged

    '''

    df_messages = pd.read_csv(messages_filename)
    df_categories = pd.read_csv(categories_filename)

    return pd.merge(df_messages, df_categories, on='id')


def clean_data(df):
    '''
    Clean the data
    
    Args:
        df (pandas.DataFrame): dataframe containing the uncleaned data
    
    Returns:
        df (pandas.DataFrame): dataframe containing the cleaned data

    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    categories_names = categories[:1].applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    categories.columns = categories_names

    # convert category values to just numbers 0 or 1
    categories = categories.applymap(lambda s: int(s[-1]))
    categories['related'].replace(2, 0, inplace=True)

    # replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates and drop null values
    df.drop_duplicates(inplace=True)
    df.dropna(subset=categories_names, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    saves the dataframe to a sqllite database file

    Args:
        df (pandas.DataFrame): dataframe containing the data
        database_filename (str): database filename

    '''
    engine = create_engine('sqlite:///'+ database_filename +'.db')
    df.to_sql(TABLE_NAME, engine, index=False)
    engine.dispose()


def parse_arguments():
    '''
    Parse the command line arguments

    Returns:
        categories_filename (str)
        messages_filename (str)
        database_filename (str) 
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Process Data", prefix_chars='-+')
    parser.add_argument('-m', '--messages-filename', type = str, default = MESSAGES_DEFAULT_FILENAME, help = 'Messages dataset filename')
    parser.add_argument('-c', '--categories-filename', type = str, default = CATEGORIES_DEFAULT_FILENAME, help = 'Categories dataset filename')
    parser.add_argument('-d', '--database-filename', type = str, default = DATABASE_DEFAULT_FILENAME, help = 'Database filename')
    args = parser.parse_args()

    return args.messages_filename, args.categories_filename, args.database_filename

    

def process_data(messages_filename, categories_filename, database_filename):
    '''
    Process the data and save it in a database

    Args:
        categories_filename (str)
        messages_filename (str)
        database_filename (str)
    '''

    print('Loading data ⌛...\n')
    df = load_data(messages_filename, categories_filename)

    print('Cleaning data 🧹... \n')
    df = clean_data(df)

    print('Saving the database as {} 💾...\n'.format(database_filename))
    save_data(df, database_filename)

    print('Done processing✅')


if __name__ == '__main__':
    messages_filename, categories_filename, database_filename = parse_arguments()
    process_data(messages_filename, categories_filename, database_filename)

