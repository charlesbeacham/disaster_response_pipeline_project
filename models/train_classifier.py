import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, fbeta_score, classification_report
import pickle

# instantiate global variables for tokenizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) 
pattern = re.compile(r'[^a-zA-Z0-9]')

def load_data(database_filepath):
    db = database_filepath.split('/')[-1]
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(db, engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.to_list()

    return X, Y, category_names
    

def tokenize(text):
    text = pattern.sub(' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens if w not in stop_words]    

    return tokens

def build_model():
    
    # define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(ComplementNB())),
    ])

    # cross-validation parameters
    param = {
        'vect__ngram_range': [(1,1), (1,2), (2,2)],
        'vect__max_df':[0.6, 0.8, 1.0],
        'tfidf__norm': ['l1', 'l2', None],
        'tfidf__use_idf': [True, False],
        'clf__estimator__alpha': (0.2, 0.5, 0.8, 2.0), 
    }

    # make scorer
    multi_fbeta = make_scorer(fbeta_score, beta=2, average='macro', zero_division=0)    

    # define main model
    cv = RandomizedSearchCV(pipeline,
                      param,                      
                      verbose=5,
                      error_score='raise',
                      return_train_score=False,
                      n_jobs=-1,
                      n_iter=25,
                      scoring=multi_fbeta,
                      random_state=41,
                     )
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    multi_accuracy = make_scorer(accuracy_score)
    multi_precision = make_scorer(precision_score, average='macro', zero_division=0)
    multi_recall = make_scorer(recall_score, average='macro', zero_division=0)
    
    accuracy = multi_accuracy(model, X_test, Y_test)
    precision = multi_precision(model, X_test, Y_test)
    recall = multi_recall(model, X_test, Y_test)
    
    print(f'macro-averaged accuracy: {accuracy:.2f} -- macro-averaged precision: {precision:.2f} -- macro-averaged recall: {recall:.2f}')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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