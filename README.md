# Disaster Response Pipeline Project

Summary of Project:

Use text messages sent during natural disasters to train a machine learning model that can be used to classify messages according to the category or categories for which the message is most related.  This can help responders filter to just the messages that are the most important for them to read.

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

See spec-file.txt for full list of packages used.  The package manager used was Anaconda and coded in Python 3.11.7.  THe main packages used were pandas, sci-kit learn, and flask.

You can use the spec-file to create an exact anaconda virtual environment with the command `conda create --name myenv --file spec-file.txt`

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://127.0.0.1:3001/ or the local host link as directed by the command prompt.


## Project Motivation<a name="motivation"></a>

For this project I was interested in demonstrating the following:
* Load raw data stored in csv files into a dataframe
* Create a cleansed dataframe that can be stored in a local database file
* Load that data into a dataframe and use it train a scikit-learn classifier
* Save the trained model to a pickle file
* Use the trained model in a web app to classify new data and to display some descriptive information about the data with some charts

All of this is done via interacting with the command line.  See instructions below for more info.

## File Descriptions <a name="files"></a>

- data/process_data.py - python script that takes the raw csv data and saves the cleansed data in a local database file.
- models/train_classifier.py - python script that reads the cleansed dataframe from the local database and returns a fully trained classification model which is saved as a pickle (.pkl) file
- app/run.py - python script that loads the flask web app
- spec-file.txt - explicit list of packages used to create the virtual environment.  Created with `conda list --explicit > spec-file.txt`
- ETL Pipeline Preparation.ipynb - notebook used as interim step when creating process_data.py.  Not needed for main analysis.
- ML Pipeline Preparation.ipynb - notebook used as interim step when creating train_classifier.py.  Not needed for main analysis.


## Results<a name="results"></a>

The main classifier metrics are printed to the command line.  The new classifications and visuals can be accessed via the Flask webapp.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The raw data .csv files are from the company Figure 8.  Please feel free to use the code here as you like.