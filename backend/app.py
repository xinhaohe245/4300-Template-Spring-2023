import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import similarity as sim
from dotenv import load_dotenv, find_dotenv

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


# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search(itemname):
    query_sql = f"SELECT * FROM fast_food_items"
    keys = ["restaurant", "item", "calories", "total_fat", "cholesterol", "sodium", "protein"]
    data = mysql_engine.query_selector(query_sql)
    results = [dict(zip(keys, i)) for i in data]
    top_10 = sim.edit_distance_sim(itemname, results)
    return json.dumps(top_10)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/food")
def food_search():
    text = request.args.get("food")
    return sql_search(text)

# app.run(debug=True)