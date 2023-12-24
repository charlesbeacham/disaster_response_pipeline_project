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
    '''
      Read a database file into a dataframe and create X and Y varibles along with category 
      names that will be used in training.

            Parameters:
                    database_filepath: a string with the filepath to a local database file.
            Returns:
                    X: an X matrix that can be used for machine learning
                    Y: a Y matrix of target variables that can be used for machine learning
                    category_names: a list of Y variables column names.
    '''
    db = database_filepath.split('/')[-1]
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(db, engine) # read cleaned data into dataframe

    # split data into an X and Y variable
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.to_list()

    return X, Y, category_names
    

def tokenize(text):
    '''
      A custom text tokenizer to be used as the tokeinzer in an sklearn CountVectorizer object.
      It takes in a text string and removes any character that is not a number or letter,
      creates a token for each word, and then lemmatizes the tokens and finally removes any 
      stop words.  The lower case function is handled inside the CountVectorizer itself in 
      the build_model function (via lowercase=True).

            Parameters:
                    text: A string sentence to be tokenized.
            Returns:
                    tokens: a list containing the tokens.
    '''
    text = pattern.sub(' ', text) # remove any character not a letter or number
    tokens = word_tokenize(text) # create a list of word tokents

    # remove stop words and lemmatize each word
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens if w not in stop_words]    

    return tokens

def build_model():
    '''
      Construct a scikit-learn pipeline classification model that can be used for training.
      Also build a custom scorer and the parameter grid to be searched over in order to tune
      the hyper-parameters.
            Parameters:
                    none
            Returns:
                    cv: An sklearn classification model
        '''
    
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
    '''
      Print the measurement values of the test data.  Both the macro-averaged values and
      the precision, recall, and f1_score by each target category.

            Parameters:
                    model: a trained sklearn classification model
                    X_test: an array of the X-matrix created with train_test_split used for making prediction
                    Y_test: an array of the true answers to compare against the X_test.
                    category_names: a list of the target variable names

            Returns:
                    nothing, but prints the results to the command line.
    '''
    

    # print the macro-averaged accuracy, precision, and recall
    multi_accuracy = make_scorer(accuracy_score)
    multi_precision = make_scorer(precision_score, average='macro', zero_division=0)
    multi_recall = make_scorer(recall_score, average='macro', zero_division=0)
    
    accuracy = multi_accuracy(model, X_test, Y_test)
    precision = multi_precision(model, X_test, Y_test)
    recall = multi_recall(model, X_test, Y_test)
    
    print(f'macro-averaged accuracy: {accuracy:.2f} -- macro-averaged precision: {precision:.2f} -- macro-averaged recall: {recall:.2f}')

    # code below will print the precision, recall, and f1_score for each target
    pred = pd.DataFrame(model.predict(X_test))
    target_names = category_names

    # enumerate over each category to print the information from sklearn's classification report
    for i, col in enumerate(Y_test):
        print(f'-------{target_names[i]}-{i}-------\n')
        measure_dict = classification_report(Y_test[col], pred[i], output_dict=True)
        
    # print the value if it exists for predicting 1.  If there are no 1 values, return 0 to prevent an error.
        try:
            precision = measure_dict['1']['precision']
        except:
            precision = 0.
    
        try:
            recall = measure_dict['1']['recall']
        except:
            recall = 0.
    
        try:
            f1_score = measure_dict['1']['f1-score']
        except:
            f1_score = 0.
            
    
        print(f'precision: {precision:0.2}    recall: {recall:0.2}    f1-score: {f1_score:0.2}')


def save_model(model, model_filepath):
    '''
      Save a trained scikit-learn classification model as a pickle file that can be used later.

            Parameters:
                    model: a trained sklearn classification model to be saved.
                    model_filepath: string of filepath for location to save the model.

            Returns:
                    nothing, but the file is saved.
    '''
    pickle.dump(model, open(model_filepath, 'wb')) # save the trained model


def main():
    '''
      This is the main function that runs all the code based on the inputs in the command line.
      Should be of the form similar to:
      python ./models/train_classifier.py ./data/DisasterResponse.db classifier.pkl
    '''
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