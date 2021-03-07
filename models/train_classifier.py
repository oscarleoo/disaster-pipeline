import sys
import nltk
import pickle
nltk.download('punkt')

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Loads the sql data
    --
    Inputs:
        database_filepath: The path to the data
    Outputs:
        X: the messages that we want to train on
        Y: The categories
        columns: The categoriy names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_data', con=engine) 
    X, Y = df.message, df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, list(Y.columns)


def tokenize(text):
    """
    Tokenizes text in preparation for training
    --
    Inputs:
        text: The text to tokenize
    Outputs:
        texts: The tokenized text
    """
    return word_tokenize(text.lower())


def build_model():
    """
    Function for building the model using pipeline and gridsearch
    --
    Outputs:
        model: A GridSearchCV containing the model and parameters we want to search for 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return GridSearchCV(pipeline, param_grid={
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    })


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating the model using classification report and prints the results
    --
    Inputs:
        model: The model to evaluate
        X_test: The input data to make prediction on
        Y_test: The correct answers
        category_names: A list with the names of the categories
    """
    P_test = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(classification_report(Y_test[c], P_test[:, i]))


def save_model(model, model_filepath):
    """
    Function for saving the model
    --
    Inputs:
        model: The model we want to save
        model_filepath: The place to save the model
    """
    with open(model_filepath, 'wb') as file:
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