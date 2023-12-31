import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = joblib.load(r"./classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # Question 1: what are the counts for each category?
    values = df.drop(columns=['id', 'message', 'original', 'genre']) # create df of only values
    values = values.loc[values['related']==1].drop(columns=['related']) # only look at related messages and drop related
    value_sum = values.sum() # sum to get counts for each category
    fig_1 = px.bar(value_sum, x=value_sum.index, y=value_sum.values,
                  title="Count of Messages by Category",
                  template="simple_white",
            
            )

    fig_1.update_layout(xaxis_title="Category", yaxis_title="Count")
    fig_1.update_xaxes(tickangle=-90)

    # for earthquakes, what categories are most correlated?
    earthquake = values.loc[values['earthquake']==1]
    earthquake_sum = earthquake.sum().sort_values(ascending=False).drop('earthquake')
    
    fig_2 = px.bar(earthquake_sum, x=earthquake_sum.index, y=earthquake_sum.values,
                   title="Most Common Message Categories Related to Earthquakes",
                   template="simple_white",
                
                )
    
    fig_2.update_layout(xaxis_title="Category", yaxis_title="Count")
    fig_2.update_xaxes(tickangle=-90)

        
    

    
    graphs = [fig_1, fig_2]
    
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()