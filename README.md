# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

See requirements.txt for full list of packages used.  The package manager used was Anaconda and coded in Python 3.11.4.  THe main packages used were numpy, pandas, and sci-kit learn.

## Project Motivation<a name="motivation"></a>

For this project, I was interested in answering some business questions related to the Seattle Airbnb data.  The specific questions explored were:

1. What is the average review score by neighborhood?
2. Which features are the most important when it comes to determining which Airbnbs are more popular? (i.e. have the most stays, specifically the highest revews_per_month)
3. What months of the year are most popular for visiting Seattle?

## File Descriptions <a name="files"></a>

The notebook analysis file used to answer all 3 questions is airbnb.ipynb.  In addition:

- listings.csv - raw data that displays information realted to the listings
- calendar.csv - raw data that displays availability per listing
- reviews.csv - raw data that displays all the comments that reviewers left per listing
- requirements.txt - full list of packages used

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@cbeacham_96550/before-that-trip-to-seattle-2af2f609c8cb).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The raw data .csv files are from kaggle [here](https://www.kaggle.com/datasets/airbnb/seattle).  Any code from the web that was used as reference has been linked in the notebook file.  Please feel free to use the code here as you like.