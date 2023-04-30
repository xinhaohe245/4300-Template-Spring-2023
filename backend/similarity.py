import numpy as np

def cosine_sim(query_vec, docs, results, ratings):
  scores = docs.dot(query_vec.T).flatten()
  top_50 = np.argsort(-scores)[:51][1:]
  restaurants = [results[i] for i in top_50]
  for restaurant in restaurants:
    restaurant['avg_rating'] = ratings.get(restaurant['restaurant'])
  restaurants.sort(key=lambda x : -x['avg_rating'])
  return restaurants[:10]





  