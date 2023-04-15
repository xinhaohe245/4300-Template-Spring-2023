import numpy as np

def cosine_sim(query_vec, docs, results):
  scores = docs.dot(query_vec.T).flatten()
  top_10 = np.argsort(-scores)[:11]
  return [results[i] for i in top_10[1:]]



  