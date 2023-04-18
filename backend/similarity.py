import numpy as np

def cosine_sim(query_vec, docs, results, ratings):
  scores = docs.dot(query_vec.T).flatten()
  top_30 = np.argsort(-scores)[:31]
  restaurants = [results[i] for i in top_30[1:]]
  for restaurant in restaurants:
    rating = ratings.get(f"\"{restaurant['restaurant']}\"", 0)
    restaurant['avg_rating'] = rating
  restaurants.sort(key=lambda x : -x['avg_rating'])
  return restaurants[:10]

  # return [results[i] for i in top_10[1:]]



  