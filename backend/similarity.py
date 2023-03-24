from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

n_features = 200

tfidf_vec =  TfidfVectorizer(max_features = n_features, stop_words = "english", 
norm='l2')

def cosine_similarity(query, results):
  combined = [query] + [result['item'] for result in results]
  doc_by_term = tfidf_vec.fit_transform(combined).toarray()
  query_vec = doc_by_term[0]
  result_vecs = doc_by_term[1:]
  scores = result_vecs.dot(query_vec)
  rankings = np.argsort(-scores)
  top_10 = [results[i] for i in rankings[:10]]
  return top_10


  