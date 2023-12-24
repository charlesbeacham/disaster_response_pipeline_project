import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories,
                  how='inner', on='id')
    
    return df

def clean_data(df):
    '''
      Takes in the raw merged message and category data and returns a dataframe that 
      can be used for machine learning.  The categories are split into columns so that
      each category can be trained independently. 
            Parameters:
                    df: pandas dataframe to be cleaned
            Returns:
                    df: cleaned dataframe to be used for machine learning
    '''
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = list(categories.iloc[0].apply(lambda x: x[:-2]))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.loc[df['related']==2, 'related'] = 1 # force 2 to equal the same as a 1
    df.drop(df.loc[df.duplicated()].index, inplace=True)

    return df

def save_data(df, database_filename):
    '''
      Takes the cleaned dataframe and saves it to a local database file.
            Parameters:
                    df: pandas dataframe
                    database_filename: string filepath for where to store database
            Returns:
                    nothing but database is saved
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)  


def main():
    '''
      This is the main function that runs all the code based on the inputs in the command line.
      Should be of the form similar to:
      python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db 
    '''
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