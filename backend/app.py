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

food_query = "select restaurant, item_name, item_description, calories, cholesterol, sodium from fast_food_items"
keys = ["restaurant", "item_name", "item_description", "calories", "cholesterol", "sodium"]
data = mysql_engine.query_selector(food_query)
results = [dict(zip(keys, i)) for i in data]

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.8, min_df=100)
td_matrix = vectorizer.fit_transform([result['item_description'] for result in results])
docs_compressed, s, words_compressed = svds(td_matrix, k=50)
words_compressed = normalize(words_compressed.T, axis=1)
docs_compressed = normalize(docs_compressed)

rating_query = "select restaurant, avg(rating) as avg_rating from reviews group by restaurant"
rating_data = mysql_engine.query_selector(rating_query)
rating_results = [dict(zip(['restaurant', 'avg_rating'], i)) for i in rating_data]
restaurant_ratings = {i['restaurant'] : float(round(i['avg_rating'], 2)) for i in rating_results}

review_query = "select restaurant, group_concat(review) as all_reviews from reviews group by restaurant"
review_data = mysql_engine.query_selector(review_query)
review_results = [dict(zip(['restaurant', 'all_reviews'], i)) for i in review_data]
restaurant_reviews = {i['restaurant'] : i['all_reviews'] for i in review_results}

def sql_search(query, restaurant_filter = None):
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = query_tfidf.dot(words_compressed)
    top_10 = sim.cosine_sim(query_vec, docs_compressed, results, restaurant_ratings, restaurant_filter)
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
