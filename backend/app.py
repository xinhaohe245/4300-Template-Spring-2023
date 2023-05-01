import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import similarity as sim
from dotenv import load_dotenv, find_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.metrics import edit_distance
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds


# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to

load_dotenv(find_dotenv())

MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_USER_PASSWORD = os.getenv('MYSQL_USER_PASSWORD')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DATABASE = "fastfooddb"

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

query = "select restaurant, item_name, item_description, calories, cholesterol, sodium from fast_food_items"
keys = ["restaurant", "item_name", "item_description", "calories", "cholesterol", "sodium"]
data = mysql_engine.query_selector(query)
results = [dict(zip(keys, i)) for i in data]

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.8, min_df=100)
td_matrix = vectorizer.fit_transform([result['item_description'] for result in results])
docs_compressed, s, words_compressed = svds(td_matrix, k=50)
words_compressed = normalize(words_compressed.T, axis=1)
docs_compressed = normalize(docs_compressed)

query = "select restaurant, avg(rating) as avg_rating from reviews group by restaurant"
keys = ['restaurant', 'avg_rating']
data = mysql_engine.query_selector(query)
rating_results = [dict(zip(keys, i)) for i in data]
restaurant_ratings = {i['restaurant'] : float(round(i['avg_rating'], 2)) for i in rating_results}

query = "select restaurant, group_concat(review) as all_reviews from reviews group by restaurant"
keys = ['restaurant', 'all_reviews']
data = mysql_engine.query_selector(query)
review_results = [dict(zip(keys, i)) for i in data]
restaurant_to_index = {item['restaurant'] : i for i, item in enumerate(review_results)}

rest_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.8, min_df=1)
td_matrix = rest_vectorizer.fit_transform([result['all_reviews'] for result in review_results])
rest_docs, s, rest_words = svds(td_matrix, k=30)
rest_words = normalize(rest_words.T, axis=1)
rest_docs = normalize(rest_docs)

def sql_search(query, restaurant_filter = None):
    rest_vec = None
    if restaurant_filter != 'null':
        restaurants = restaurant_filter.split(',')
        rest_vec = np.zeros(s.shape)
        for restaurant in restaurants:
            rest_vec += rest_docs[restaurant_to_index[restaurant]]
        rest_vec /= len(restaurants)

    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = query_tfidf.dot(words_compressed)
    top_10 = sim.cosine_sim(query_vec, docs_compressed, results, 
    restaurant_ratings, restaurant_to_index, rest_docs, rest_vec)
    return json.dumps(top_10)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/food")
def food_search():
    text = request.args.get("food")
    restaurant = request.args.get("restaurant")
    return sql_search(text, restaurant)

# app.run(debug=True)
