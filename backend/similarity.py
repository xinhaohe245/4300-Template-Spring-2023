from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.metrics import edit_distance

n_features = 2000

tfidf_vec = TfidfVectorizer(max_features = n_features, stop_words = "english", 
max_df = 0.8, min_df=10, norm='l2')

def cosine_similarity(query, results):
  combined = [query] + [result['item'] for result in results]
  doc_by_term = tfidf_vec.fit_transform(combined).toarray()
  query_vec = doc_by_term[0]
  result_vecs = doc_by_term[1:]
  scores = result_vecs.dot(query_vec)
  rankings = np.argsort(-scores)
  if scores[rankings[0]] < 0.2:
    return edit_distance(query, results)
  top_10 = [results[i] for i in rankings[:10]]
  return top_10

def edit_distance(query, results):
  scores = np.empty(len(results))
  for i, result in enumerate([result['item'] for result in results]):
    scores[i] = edit_distance(query, result)
  rankings = np.argsort(scores)
  top_10 = [results[i] for i in rankings[:10]]
  return top_10


  