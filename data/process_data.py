import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loading and compining the messages and categories csv files
    --
    Inputs:
        messages_filepath: csv file contains messages
        categories_filepath: csv file contains categories
    Outputs:
        df: the combined dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id', how='outer')


def clean_data(df):
    """
    Preprocess the combined data in preperation for training
    This includes making creating binary targets of the categories,
    and removing duplicates
    
    I'm also removing the instances where related is 2 because I couldn't find an explanation of what that means.
    --
    Inputs:
        df: messages and categories combined dataframe dataframe
    Outputs:
        df: cleaned dataframe
    """ 
    categories = df.categories.str.split(';', expand=True)
    category_names = [c[:-2] for c in categories.iloc[0]]
    categories.columns = category_names

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    categories = categories[categories.related != 2]
        
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    return df.dropna().drop_duplicates()
    

def save_data(df, database_filename):
    """
    Saves the dataframe into a sqlite database 
    --
    Inputs:
        df: messages and categories combined dataframe dataframe
        database_filename: Name of the database file
    """    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_data', engine, index=False)  


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