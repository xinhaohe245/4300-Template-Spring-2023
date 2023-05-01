import numpy as np

def cosine_sim(query_vec, docs, results, ratings, restaurant_filter):
  scores = docs.dot(query_vec.T).flatten()
  top_50 = np.argsort(-scores)[:51][1:]
  top_items = [results[i] for i in top_50]

  if restaurant_filter and restaurant_filter != 'null':
    top_items = [item for item in top_items if item['restaurant'] == restaurant_filter]

  for item in top_items:
    item['avg_rating'] = ratings.get(item['restaurant'])
  top_items.sort(key=lambda x : -x['avg_rating'])
  return top_items[:10]





